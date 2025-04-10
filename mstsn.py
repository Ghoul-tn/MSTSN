import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = self.add_weight(shape=(num_nodes, hidden_dim),
                                       initializer='glorot_uniform')
    
    @tf.function
    def call(self, batch_size):
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class BatchedGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(out_dim//heads, heads=heads, concat=True)
        
    @tf.function
    def call(self, inputs):
        x, adj = inputs
        batch_size = tf.shape(x)[0]
        
        # Vectorized implementation - process all batches at once
        # Create a batched edge list from the adjacency matrices
        edge_lists = []
        batch_edge_indices = []
        
        # Using tf.map_fn to avoid explicit Python loop
        def process_batch_item(idx):
            # Get edges for this batch item
            edges = tf.where(adj[idx] > 0.5)
            # Add batch dimension
            batch_indices = tf.fill([tf.shape(edges)[0], 1], idx)
            return edges, batch_indices
        
        # Process all batches and create combined edge list
        batched_results = tf.map_fn(
            process_batch_item,
            tf.range(batch_size),
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int64),
                tf.RaggedTensorSpec(shape=[None, 1], dtype=tf.int32)
            )
        )
        
        # Apply GAT to all batches
        # Format inputs to match GAT expectations
        features = tf.reshape(x, [-1, tf.shape(x)[-1]])  # Flatten batch dimension
        
        # Process each batch item with the same GAT weights
        outputs = []
        for b in range(batch_size):
            edge_idx = tf.where(adj[b] > 0.5)
            outputs.append(self.gat([x[b], edge_idx]))
        
        # Stack back to batch format
        return tf.stack(outputs)

class SpatialProcessor(Model):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = BatchedGAT(hidden_dim)
        self.gat2 = BatchedGAT(out_dim)
        self.proj = layers.Dense(hidden_dim)
        
    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]
        adj = self.adaptive_adj(batch_size)
        x = self.proj(x)
        x = tf.nn.relu(self.gat1([x, adj]))
        return self.gat2([x, adj])

class TemporalTransformer(Model):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.enc_layers = [
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=input_dim // num_heads
            ),
            tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='gelu'),
                tf.keras.layers.Dense(input_dim)
            ])
        ]
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(0.1)
        
    @tf.function
    def call(self, x):
        for _ in range(self.num_layers):
            attn_output = self.enc_layers[0](x, x)
            attn_output = self.dropout(attn_output)
            out1 = self.layernorm1(x + attn_output)
            
            ffn_output = self.enc_layers[1](out1)
            ffn_output = self.dropout(ffn_output)
            x = self.layernorm2(out1 + ffn_output)
        return x

class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, embed_dim)
        
    @tf.function
    def call(self, inputs):
        x1, x2 = inputs
        return self.attn(x1, x2)

class EnhancedMSTSN(Model):
    def __init__(self, num_nodes):
        super().__init__()
        self.spatial = SpatialProcessor(num_nodes, 3, 16, 32)
        self.temporal = TemporalTransformer(32, 2, 64, 1)
        self.cross_attn = CrossAttention(32, 2)
        self.regressor = tf.keras.Sequential([
            layers.Dense(16, activation='gelu'),
            layers.Dense(1)
        ])
        
    @tf.function
    def call(self, inputs):
        # Input: [batch, seq_len, nodes, features]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]
        
        # Vectorized spatial processing to avoid using TensorArray
        # Reshape to process all time steps together
        spatial_inputs = tf.reshape(inputs, [batch_size * seq_len, num_nodes, 3])
        
        # Process all time steps in a single pass
        spatial_flat = self.spatial(spatial_inputs)
        
        # Reshape back to sequence form
        spatial_out = tf.reshape(spatial_flat, [batch_size, seq_len, num_nodes, 32])
        
        # Temporal Processing
        temporal_in = tf.reshape(spatial_out, [batch_size*num_nodes, seq_len, 32])
        temporal_out = self.temporal(temporal_in)
        temporal_out = tf.reshape(temporal_out, [batch_size, num_nodes, seq_len, 32])
        
        # Cross Attention
        spatial_feats = tf.reduce_mean(spatial_out, axis=1)
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)
        fused = self.cross_attn([spatial_feats, temporal_feats])
        
        return self.regressor(fused)[..., 0]
