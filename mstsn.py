import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import scipy.sparse

class GraphAttention(layers.Layer):
    def __init__(self, output_dim, heads=1, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.query_dense = layers.Dense(output_dim * heads)
        self.key_dense = layers.Dense(output_dim * heads)
        self.value_dense = layers.Dense(output_dim * heads)
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs, adj_matrix, training=False):
        # inputs: [batch_size, num_nodes, input_dim]
        # adj_matrix: [num_nodes, num_nodes]
        if not isinstance(adj_matrix, tf.Tensor):
            adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)        
        # Apply linear transformations
        queries = self.query_dense(inputs)  # [batch_size, num_nodes, output_dim*heads]
        keys = self.key_dense(inputs)       # [batch_size, num_nodes, output_dim*heads]
        values = self.value_dense(inputs)   # [batch_size, num_nodes, output_dim*heads]
        
        # Reshape for multi-head attention
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        queries = tf.reshape(queries, [batch_size, num_nodes, self.heads, self.output_dim])
        keys = tf.reshape(keys, [batch_size, num_nodes, self.heads, self.output_dim])
        values = tf.reshape(values, [batch_size, num_nodes, self.heads, self.output_dim])
        
        # Calculate attention scores
        # [batch_size, heads, num_nodes, num_nodes]
        attention_scores = tf.einsum('bihd,bjhd->bhij', queries, keys)
        
        # Scale attention scores
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.output_dim, tf.float32))
        
        # Apply adjacency matrix as mask
        # [1, 1, num_nodes, num_nodes]
        adj_mask = tf.expand_dims(tf.expand_dims(adj_matrix, 0), 0)
        
        # Set attention scores to -inf where there are no connections
        mask = tf.cast(tf.equal(adj_mask, 0), dtype=tf.float32) * -1e9
        attention_scores = attention_scores + mask
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Apply attention weights to values
        # [batch_size, heads, num_nodes, output_dim]
        attention_output = tf.einsum('bhij,bjhd->bihd', attention_weights, values)
        
        # Combine heads
        attention_output = tf.reshape(attention_output, [batch_size, num_nodes, self.heads * self.output_dim])
        
        return attention_output

class SpatialProcessor(layers.Layer):
    def __init__(self, num_nodes, output_dim, adj_matrix):
        super().__init__()
        # Initialize projection layer for residual connection
        self.projection = layers.Dense(output_dim)  # Add this line
        
        # Rest of existing code remains the same
        if scipy.sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)        
        if adj_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(f"Adjacency matrix shape {adj_matrix.shape} doesn't match num_nodes {num_nodes}")        
        self.adj_matrix = tf.constant(
            self.normalize_adjacency(adj_matrix),
            dtype=tf.float32
        )
        self.gat1 = GraphAttention(output_dim // 2, heads=4)
        self.gat2 = GraphAttention(output_dim, heads=1)
        self.dropout = layers.Dropout(0.2)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def normalize_adjacency(self, adj_matrix):
        """Row-normalize adjacency matrix with proper shape checks"""
        # Ensure 2D array conversion
        if scipy.sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        
        # Validate matrix dimensions
        if adj_matrix.ndim != 2:
            raise ValueError(f"Adjacency matrix must be 2D, got {adj_matrix.ndim}D input")
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adj_matrix.shape}")
    
        # Calculate degree matrix
        degree = np.sum(adj_matrix, axis=1, keepdims=True)  # Keep dimensions for broadcasting
        return adj_matrix / (degree + 1e-6)

    def call(self, inputs, training=False):
        # Project inputs to match GAT output dimension
        
        projected_inputs = self.projection(inputs)  # Add this line
        
        x = self.gat1(inputs, self.adj_matrix, training=training)
        x = self.dropout(tf.nn.relu(x), training=training)
        x = self.gat2(x, self.adj_matrix, training=training)
        
        # Add projected inputs instead of original inputs
        return self.layer_norm(x + projected_inputs)  # Modified this line
        
class TemporalTransformer(layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout_rate=0.2):  # Increased dropout
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, 32)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(32)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
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
    def __init__(self, num_nodes, adj_matrix):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial = SpatialProcessor(num_nodes, 32, adj_matrix)
        self.temporal = TemporalTransformer(num_heads=2, ff_dim=64)
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=2, 
            key_dim=32,
            kernel_initializer='glorot_uniform'  # Explicit initialization
        )
        self.final_dense = layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        # Add small dropout for regularization
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, training=False):
        # Get static shapes where possible
        batch_size = tf.shape(inputs)[0]
        
        # Use tensor transformation ops
        spatial_in = tf.reshape(inputs, [-1, self.num_nodes, 3])
        
        # Spatial processing
        spatial_out = self.spatial(spatial_in, training=training)
        
        # Use static shapes for reshape where possible
        spatial_out = tf.reshape(spatial_out, [batch_size, -1, self.num_nodes, 32])
        
        # Avoid excessive transpose/reshape operations
        # Use einsum instead
        temporal_in = tf.transpose(spatial_out, [0, 2, 1, 3])
        temporal_in = tf.reshape(temporal_in, [batch_size * self.num_nodes, -1, 32])
        
        temporal_out = self.temporal(temporal_in, training=training)
        
        # Use static reshaping
        temporal_out = tf.reshape(temporal_out, [batch_size, self.num_nodes, -1, 32])
        
        # Use reduce_mean with explicit axis
        spatial_feats = tf.reduce_mean(spatial_out, axis=2)
        temporal_feats = tf.reduce_mean(temporal_out, axis=2)
        
        # Use cross attention
        fused = self.cross_attn(spatial_feats, temporal_feats)
        fused = self.layernorm(fused)
        fused = self.dropout(fused, training=training)
        
        # Return final output
        return tf.squeeze(self.final_dense(fused), axis=-1)
