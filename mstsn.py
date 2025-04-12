import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.embeddings = self.add_weight(
            shape=(num_nodes, hidden_dim),
            initializer='glorot_uniform',
            name='node_embeddings'
        )
        
    def call(self):
        """Generate adjacency matrix with top-k connections"""
        # Normalize embeddings
        norm_emb = tf.math.l2_normalize(self.embeddings, axis=-1)
        
        # Compute similarity matrix
        sim_matrix = tf.matmul(norm_emb, norm_emb, transpose_b=True)
        
        print(f"Debug - sim_matrix shape: {sim_matrix.shape}")
        print(f"Debug - num_nodes: {self.num_nodes}")
        
        # Get top-k values and indices
        k = min(20, self.num_nodes - 1)  # Make sure k is valid
        print(f"Debug - k value: {k}")
        
        # Use tf.math.top_k properly
        top_k = tf.math.top_k(sim_matrix, k=k)
        
        print(f"Debug - top_k.values shape: {top_k.values.shape}")
        print(f"Debug - top_k.indices shape: {top_k.indices.shape}")
        
        # Return as separate tensors
        return top_k.values, top_k.indices

class SpatialProcessor(layers.Layer):
    def __init__(self, num_nodes, gat_units):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_units = gat_units
        self.adj_layer = AdaptiveAdjacency(num_nodes, 16)
        self.gat1 = GATConv(gat_units, attn_heads=4, concat_heads=True)
        self.gat2 = GATConv(gat_units, attn_heads=1, concat_heads=False)
        self.dropout = layers.Dropout(0.1)

    def build(self, input_shape):
        # Build the GATConv layers with appropriate shapes
        self.gat1.build([(self.num_nodes, input_shape[-1]), (None, 2)])
        self.gat2.build([(self.num_nodes, self.gat_units * 4), (None, 2)])  # 4 heads
        super().build(input_shape)

    def call(self, inputs):
        # Get adjacency matrix values and indices
        adj_values, adj_indices = self.adj_layer()
        
        # Generate edge indices for GATConv
        # Create source indices for each node (repeat each node index k times)
        source_nodes = tf.repeat(tf.range(self.num_nodes, dtype=tf.int32), repeats=adj_indices.shape[1])
        
        # Flatten the destination indices
        target_nodes = tf.reshape(adj_indices, [-1])
        
        # Stack to create edge_index tensor [2, num_edges]
        edge_indices = tf.stack([source_nodes, target_nodes], axis=1)
        
        # Add self-loops
        self_loops = tf.stack([
            tf.range(self.num_nodes, dtype=tf.int32),
            tf.range(self.num_nodes, dtype=tf.int32)
        ], axis=1)
        edge_indices = tf.concat([edge_indices, self_loops], axis=0)
        
        # Apply GAT layers
        x = self.gat1([inputs, edge_indices])
        x = tf.nn.relu(x)
        x = self.dropout(x)
        return self.gat2([x, edge_indices])
        
class TemporalTransformer(layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, 32)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(32)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)

class EnhancedMSTSN(Model):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial = SpatialProcessor(num_nodes, 32)
        self.temporal = TemporalTransformer(num_heads=2, ff_dim=64)
        self.cross_attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)
        self.final_dense = layers.Dense(1)
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Reshape inputs to process spatial features for all sequences and batches together
        spatial_in = tf.reshape(inputs, [-1, self.num_nodes, 3])
        
        # Spatial processing
        spatial_out = self.spatial(spatial_in)
        
        # Reshape back to separate batch and sequence dimensions
        spatial_out = tf.reshape(spatial_out, [batch_size, seq_len, self.num_nodes, 32])
        
        # Temporal processing - handle each node separately
        # Reshape to [batch*num_nodes, seq_len, features]
        temporal_in = tf.reshape(
            tf.transpose(spatial_out, [0, 2, 1, 3]), 
            [batch_size * self.num_nodes, seq_len, 32]
        )
        
        # Apply temporal transformer
        temporal_out = self.temporal(temporal_in, training=training)
        
        # Reshape back to [batch, num_nodes, seq_len, features]
        temporal_out = tf.reshape(
            temporal_out, 
            [batch_size, self.num_nodes, seq_len, 32]
        )
        
        # Cross-attention between spatial and temporal features
        # Mean across sequence dimension for spatial features
        spatial_feats = tf.reduce_mean(spatial_out, axis=1)  # [batch, num_nodes, features]
        
        # Mean across sequence dimension for temporal features
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)  # [batch, num_nodes, features]
        
        # Apply cross-attention
        fused = self.cross_attn(spatial_feats, temporal_feats)
        fused = self.layernorm(fused)
        
        # Final prediction
        node_preds = self.final_dense(fused)  # [batch, num_nodes, 1]
        
        # Remove last dimension to match target shape [batch, num_nodes]
        return tf.squeeze(node_preds, axis=-1)
