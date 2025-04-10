import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = self.add_weight(shape=(num_nodes, hidden_dim),
                                       initializer='glorot_uniform')
    
    def call(self, batch_size):
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class BatchedGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(out_dim//heads, heads=heads, concat=True)
        
    def call(self, inputs):
        x, adj = inputs
        batch_size = tf.shape(x)[0]
        outputs = []
        for b in range(batch_size):
            edge_idx = tf.where(adj[b] > 0.5)  # Threshold for adjacency
            outputs.append(self.gat([x[b], edge_idx]))
        return tf.stack(outputs)

class SpatialProcessor(Model):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = BatchedGAT(hidden_dim)
        self.gat2 = BatchedGAT(out_dim)
        self.proj = layers.Dense(hidden_dim)
        
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
            layers.TransformerEncoderLayer(
                num_heads=num_heads,
                intermediate_dim=ff_dim,
                dropout=0.1,
                activation='gelu',
                norm_first=True
            ) for _ in range(num_layers)
        ]
        self.norm = layers.LayerNormalization()
        
    def call(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        return self.norm(x)

class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, embed_dim)
        
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
        
    def call(self, inputs):
        # Input: [batch, seq_len, nodes, features]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]
        
        # Spatial Processing
        spatial_out = tf.TensorArray(tf.float32, size=seq_len)
        for t in tf.range(seq_len):
            spatial_out = spatial_out.write(t, self.spatial(inputs[:, t]))
        spatial_out = tf.transpose(spatial_out.stack(), [1, 0, 2, 3])
        
        # Temporal Processing
        temporal_in = tf.reshape(spatial_out, [batch_size*num_nodes, seq_len, 32])
        temporal_out = self.temporal(temporal_in)
        temporal_out = tf.reshape(temporal_out, [batch_size, num_nodes, seq_len, 32])
        
        # Cross Attention
        spatial_feats = tf.reduce_mean(spatial_out, axis=1)
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)
        fused = self.cross_attn([spatial_feats, temporal_feats])
        
        return self.regressor(fused)[..., 0]
