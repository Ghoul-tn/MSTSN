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
    
    @tf.function
    def call(self, batch_size):
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class VectorizedGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(out_dim//heads, heads=heads, concat=True)
        
    @tf.function
    def call(self, inputs):
        x, adj = inputs
        batch_size = tf.shape(x)[0]
        
        # Process all batches using vectorized operations
        edge_indices = tf.where(adj > 0.5)
        batch_indices = edge_indices[:, 0]
        edges = edge_indices[:, 1:]
        
        # Gather features for all edges
        node_features = tf.gather_nd(x, edges)
        
        # Apply GAT (simplified for demonstration)
        outputs = self.gat([node_features, edges])
        
        # Reconstruct batch outputs
        return tf.scatter_nd(
            indices=edges,
            updates=outputs,
            shape=tf.shape(x)
        )

class SpatialProcessor(Model):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = VectorizedGAT(hidden_dim)
        self.gat2 = VectorizedGAT(out_dim)
        self.proj = layers.Dense(hidden_dim)
        
    @tf.function
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
        
    @tf.function
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
        self.spatial = SpatialProcessor(num_nodes, 3, 16, 32)
        self.temporal = EfficientTransformer(32, 2, 64)
        self.regressor = tf.keras.Sequential([
            layers.Dense(16, activation='gelu'),
            layers.Dense(1)
        ])
        
    @tf.function
    def call(self, inputs):
        # Vectorized spatial processing
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        spatial_in = tf.reshape(inputs, [batch_size*seq_len, -1, 3])
        spatial_out = tf.reshape(self.spatial(spatial_in), [batch_size, seq_len, -1, 32])
        
        # Temporal processing
        temporal_in = tf.reshape(spatial_out, [batch_size*tf.shape(spatial_out)[2], seq_len, 32])
        temporal_out = tf.reshape(self.temporal(temporal_in), [batch_size, -1, seq_len, 32])
        
        # Prediction
        return self.regressor(tf.reduce_mean(temporal_out, axis=2))[..., 0]
