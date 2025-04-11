import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import GATConv

class AdaptiveAdjacency(layers.Layer):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = self.add_weight(shape=(num_nodes, hidden_dim),
                                       initializer='glorot_uniform')
        self.num_nodes = num_nodes
    
    @tf.function
    def call(self, batch_size):
        # Normalize embeddings for cosine similarity
        norm_embed = tf.math.l2_normalize(self.embedding, axis=-1)
        
        # Create adjacency matrix from embeddings (cosine similarity)
        adj = tf.matmul(norm_embed, norm_embed, transpose_b=True)
        
        # Ensure we have a valid adjacency matrix
        # Remove self-loops and ensure connectivity
        eye = tf.eye(self.num_nodes)
        adj = adj * (1.0 - eye)  # Remove self-loops
        
        # Ensure there's at least one connection per node by using top-k
        if self.num_nodes > 1:
            k = tf.minimum(2, self.num_nodes - 1)
            values, indices = tf.math.top_k(adj, k=k)
            min_value = tf.reduce_min(values)
            adj = tf.where(adj >= min_value, adj, tf.zeros_like(adj))
        
        return tf.tile(tf.expand_dims(adj, 0), [batch_size, 1, 1])

class BatchedGAT(layers.Layer):
    def __init__(self, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(out_dim//heads, heads=heads, concat=True, 
                          attn_kernel_initializer='glorot_uniform')
        
    @tf.function
    def call(self, inputs):
        x, adj = inputs
        batch_size = tf.shape(x)[0]
        
        # Process each batch item individually
        outputs = []
        for b in range(batch_size):
            # Convert dense adjacency to edge indices
            if tf.rank(adj) == 3:  # If adj is [batch, nodes, nodes]
                edges = tf.where(adj[b] > 0.5)
            else:  # If adj is already [nodes, nodes]
                edges = tf.where(adj > 0.5)
                
            # Ensure edge indices are properly formatted for GATConv
            x_b = x[b]  # Shape [nodes, features]
            
            # Add self-loops if none exist
            if tf.shape(edges)[0] == 0:
                num_nodes = tf.shape(x_b)[0]
                indices = tf.range(num_nodes)
                self_loops = tf.stack([indices, indices], axis=1)
                edges = self_loops
                
            outputs.append(self.gat([x_b, edges]))
        
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
