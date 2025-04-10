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
        norm_emb = tf.math.l2_normalize(self.embeddings, axis=-1)
        adj = tf.matmul(norm_emb, norm_emb, transpose_b=True)
        return tf.nn.sigmoid(adj) * (1 - tf.eye(self.num_nodes))
        
class SpatialProcessor(layers.Layer):
    def __init__(self, num_nodes, gat_units):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_units = gat_units
        self.adj_layer = AdaptiveAdjacency(num_nodes, 16)
        self.gat1 = GATConv(gat_units, attn_heads=4, concat_heads=True)
        self.gat2 = GATConv(gat_units, attn_heads=1, concat_heads=False)

    def build(self, input_shape):
        # Initialize GAT layers with concrete input shape
        dummy_features = tf.keras.Input(shape=(self.num_nodes, 3))
        dummy_edges = tf.keras.Input(shape=(None, 2), dtype=tf.int32)
        
        # Force build GAT layers
        _ = self.gat1([dummy_features, dummy_edges])
        _ = self.gat2([dummy_features, dummy_edges])
        
        super().build(input_shape)

    def call(self, inputs):
        # Generate adjacency matrix
        adj = self.adj_layer()
        
        # Create edge indices with self-loops
        edge_indices = tf.where(adj > 0.5)
        edge_indices = tf.cast(edge_indices, tf.int32)
        
        # Add guaranteed self-connections
        self_loops = tf.range(self.num_nodes, dtype=tf.int32)
        self_loops = tf.stack([self_loops, self_loops], axis=1)
        edge_indices = tf.concat([edge_indices, self_loops], axis=0)
        
        # GAT processing
        x = self.gat1([inputs, edge_indices])
        return self.gat2([tf.nn.relu(x), edge_indices])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_nodes, self.gat_units)
        
class TemporalTransformer(layers.Layer):
    def __init__(self, num_heads, ff_dim):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, 32)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(32)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
    def call(self, inputs):
        x = self.layernorm1(inputs + self.attn(inputs, inputs))
        return self.layernorm2(x + self.ffn(x))

class EnhancedMSTSN(Model):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial = SpatialProcessor(num_nodes, 32)
        self.temporal = TemporalTransformer(num_heads=2, ff_dim=64)
        self.cross_attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)
        self.final_dense = layers.Dense(1)

    def build(self, input_shape):
        # Explicitly build spatial processor first
        self.spatial.build((None, self.num_nodes, 3))
        super().build(input_shape)

    def call(self, inputs):
        # Input shape: [batch, seq_len, nodes, features]
        batch_size = tf.shape(inputs)[0]
        seq_len = inputs.shape[1] or tf.shape(inputs)[1]
        
        # Spatial processing
        spatial_in = tf.reshape(inputs, [-1, self.num_nodes, 3])
        spatial_out = self.spatial(spatial_in)
        spatial_out = tf.reshape(
            spatial_out, 
            [batch_size, seq_len, self.num_nodes, 32]
        )
        
        # Temporal processing
        temporal_in = tf.reshape(
            spatial_out, 
            [batch_size * self.num_nodes, seq_len, 32]
        )
        temporal_out = self.temporal(temporal_in)
        temporal_out = tf.reshape(
            temporal_out, 
            [batch_size, self.num_nodes, seq_len, 32]
        )
        
        # Cross-attention
        spatial_feats = tf.reduce_mean(spatial_out, axis=1)
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        return self.final_dense(tf.reduce_mean(fused, axis=1))
