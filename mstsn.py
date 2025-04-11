import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = self.add_weight(
            shape=(num_nodes, hidden_dim),
            initializer='glorot_uniform',
            name='node_embeddings'
        )
        self.num_nodes = num_nodes

    def call(self, batch_size):
        # L2 normalization as per spec
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        
        # Cosine similarity adjacency matrix
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        
        # Remove self-loops and ensure connectivity
        adj = adj * (1.0 - tf.eye(self.num_nodes))
        
        # Create batch dimension
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class SpatialGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(
            out_dim // heads,
            heads=heads,
            concat=True,
            attn_kernel_initializer='glorot_uniform',
            use_mask=False  # Disable mask support
        )
        
    def call(self, inputs):
        x, adj = inputs
        x = self.gat([x, adj])
        return x

class SpatialProcessor(Model):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.proj = layers.Dense(hidden_dim)
        
        # GAT layers as per architecture spec
        self.gat1 = SpatialGAT(hidden_dim)
        self.gat2 = SpatialGAT(out_dim)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        adj = self.adaptive_adj(batch_size)
        
        # Project input features
        x = self.proj(x)  # [batch_size, num_nodes, hidden_dim]
        
        # First GAT layer with ReLU
        x = tf.nn.relu(self.gat1([x, adj]))
        
        # Second GAT layer
        return self.gat2([x, adj])

class TemporalTransformer(Model):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers=1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads, input_dim//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(input_dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        # Transformer block as per spec
        attn_output = self.attention(x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout(ffn_output))

class EnhancedMSTSN(Model):
    def __init__(self, num_nodes):
        super().__init__()
        # Spatial processor parameters from spec
        self.spatial = SpatialProcessor(
            num_nodes=num_nodes,
            in_dim=3,       # NDVI, Soil, LST features
            hidden_dim=16,
            out_dim=32
        )
        
        # Temporal processor parameters from spec
        self.temporal = TemporalTransformer(
            input_dim=32,
            num_heads=2,
            ff_dim=64
        )
        
        # Cross-attention and regressor
        self.cross_attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)
        self.regressor = tf.keras.Sequential([
            layers.Dense(16, activation='gelu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        # Input: [batch, seq_len, nodes, features]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]
        
        # Spatial processing (all time steps)
        spatial_input = tf.reshape(inputs, [-1, num_nodes, 3])
        spatial_out = self.spatial(spatial_input)
        spatial_out = tf.reshape(spatial_out, [batch_size, seq_len, num_nodes, 32])
        
        # Temporal processing
        temporal_in = tf.reshape(spatial_out, [batch_size*num_nodes, seq_len, 32])
        temporal_out = self.temporal(temporal_in)
        temporal_out = tf.reshape(temporal_out, [batch_size, num_nodes, seq_len, 32])
        
        # Cross-attention fusion
        spatial_feats = tf.reduce_mean(spatial_out, axis=1)
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        # Final prediction
        return self.regressor(fused)[..., 0]
