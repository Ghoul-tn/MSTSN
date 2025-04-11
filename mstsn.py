import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = self.add_weight(
            shape=(num_nodes, hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, batch_size):
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class VectorizedGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(out_dim//heads, heads=heads, concat=True)
        
    def build(self, input_shape):
        # Properly initialize the GAT layer
        self.gat.build([input_shape[0], (None, None)])
        super().build(input_shape)
        
    def call(self, inputs):
        x, adj = inputs
        # Convert adjacency matrix to edge index format
        edge_index = tf.where(adj > 0.5)
        # Ensure proper dtype casting
        edge_index = tf.cast(edge_index, tf.int64)
        return self.gat([x, edge_index])

class SpatialProcessor(Model):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = VectorizedGAT(hidden_dim)
        self.gat2 = VectorizedGAT(out_dim)
        self.proj = layers.Dense(hidden_dim)
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        adj = self.adaptive_adj(batch_size)
        x = self.proj(x)
        x = tf.nn.gelu(self.gat1([x, adj]))
        return self.gat2([x, adj])

class EfficientTransformer(layers.Layer):
    def __init__(self, dim, heads, ff_dim, num_layers=1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(heads, dim//heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.num_layers = num_layers
        
    def call(self, x):
        for _ in range(self.num_layers):
            attn_out = self.attention(x, x)
            x = self.layernorm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.layernorm2(x + ffn_out)
        return x

class EnhancedMSTSN(Model):
    def __init__(self, num_nodes):
        super().__init__()
        self.spatial = SpatialProcessor(num_nodes, 3, 8, 16)  # Reduced dimensions
        self.temporal = EfficientTransformer(16, 2, 32)
        self.regressor = tf.keras.Sequential([
            layers.Dense(16, activation='gelu'),
            layers.Dense(1)
        ])
        
    def build(self, input_shape):
        # Initialize with example input
        example_input = tf.ones(input_shape)
        self.call(example_input)
        super().build(input_shape)
        
    def call(self, inputs):
        # Input: [batch, seq_len, nodes, features]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]
        
        # Spatial Processing
        spatial_in = tf.reshape(inputs, [batch_size*seq_len, num_nodes, 3])
        spatial_out = self.spatial(spatial_in)
        spatial_out = tf.reshape(spatial_out, [batch_size, seq_len, num_nodes, 16])
        
        # Temporal Processing
        temporal_in = tf.reshape(spatial_out, [batch_size*num_nodes, seq_len, 16])
        temporal_out = self.temporal(temporal_in)
        temporal_out = tf.reshape(temporal_out, [batch_size, num_nodes, seq_len, 16])
        
        # Prediction
        return self.regressor(tf.reduce_mean(temporal_out, axis=2))[..., 0]
