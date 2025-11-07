# ------------------------------------------------------
# Step 1: Replace TF1 imports & random seed with TF2 equivalents
# ------------------------------------------------------
import time
import tensorflow as tf
import numpy as np
import sys
import os
import csv
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score
import scipy.io as sio
import pickle as pkl
import copy
import scipy.spatial.distance
from tqdm import tqdm
# Use Keras directly from TF2 (no separate keras import)
from tensorflow.keras import layers, Model, optimizers, initializers, metrics
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
import scipy.sparse as sp
import ABIDE_Parser as Reader

# Set random seed (TF2 equivalent of tf.set_random_seed)
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# ------------------------------------------------------
# Step 2: Remove tf.app.flags; use config dict for hyperparameters
# ------------------------------------------------------
# Replace TF1 flags with a config dictionary (easier to manage in TF2)
config = {
    'node_num': 110,          # Number of Graph nodes
    'output_dim': 1,           # Number of output dimensions
    'learning_rate': 0.0001,   # Initial learning rate for model
    'learning_rate_mask': 0.01,# Learning rate for mask optimization
    'batch_num': 10,           # Batch size (original 'batch_num' = batch size)
    'epochs': 1000,            # Epochs for model training
    'epochs_mask': 400,        # Epochs for mask optimization
    'attn_heads': 5,           # Number of attention heads
    'hidden1_gat': 24,         # GAT hidden layer 1 units
    'output_gat': 3,           # GAT output layer units
    'dropout': 0.0,            # Dropout rate (1 - keep prob)
    'in_drop': 0.0,            # Input dropout rate
    'weight_decay': 5e-4,      # L2 weight decay
    'early_stopping': 15,      # Early stopping tolerance
    'fold': 0                  # Target fold to train
}

# ------------------------------------------------------
# Step 3: Replace custom utils with TF2/Keras built-ins
# ------------------------------------------------------
def accuracy(preds, labels):
    """TF2-compatible accuracy function (matches original logic)"""
    correct_prediction = tf.equal(tf.round(preds), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ------------------------------------------------------
# Step 4: Convert TF1 gat_layer to TF2 Keras Layer subclass
# ------------------------------------------------------
class GATLayer(layers.Layer):
    def __init__(self, input_dim, F_, attn_heads=1, attn_heads_reduction='concat',
                 activation=tf.nn.relu, use_bias=True, dropout_rate=0.0, in_drop=0.0, name=''):
        super().__init__(name=f'gat_layer{name}')  # Call parent Layer __init__

        # Store hyperparameters
        self.input_dim = input_dim          # Input feature dimension per node
        self.F_ = F_                        # Output feature dimension per node
        self.attn_heads = attn_heads        # Number of attention heads
        self.attn_heads_reduction = attn_heads_reduction  # Head aggregation method
        self.activation = activation        # Activation function
        self.use_bias = use_bias            # Whether to use bias
        self.dropout_rate = dropout_rate    # Attention dropout rate
        self.in_drop = in_drop              # Input feature dropout rate

    def build(self, input_shape):
        """Build trainable weights (TF2 lazy initialization)"""
        # Use Keras built-in GlorotUniform (replaces custom glorot function)
        glorot_init = initializers.GlorotUniform(seed=seed)
        zero_init = initializers.Zeros()

        # Initialize weights for each attention head
        self.weights_list = []              # Linear transformation weights: (input_dim, F_)
        self.attn_self_weights = []         # Self-attention weights: (F_, 1)
        self.attn_neigh_weights = []        # Neighbor attention weights: (F_, 1)

        for i in range(self.attn_heads):
            # Linear kernel (replaces custom glorot([input_dim, F_]))
            w = self.add_weight(
                shape=(self.input_dim, self.F_),
                initializer=glorot_init,
                dtype=tf.float32,
                name=f'weights_{i}',
                trainable=True
            )
            # Self-attention weight (replaces custom glorot([F_, 1]))
            attn_self = self.add_weight(
                shape=(self.F_, 1),
                initializer=glorot_init,
                dtype=tf.float32,
                name=f'attn_self_weights_{i}',
                trainable=True
            )
            # Neighbor attention weight (replaces custom glorot([F_, 1]))
            attn_neigh = self.add_weight(
                shape=(self.F_, 1),
                initializer=glorot_init,
                dtype=tf.float32,
                name=f'attn_neighs_weights_{i}',
                trainable=True
            )

            self.weights_list.append(w)
            self.attn_self_weights.append(attn_self)
            self.attn_neigh_weights.append(attn_neigh)

        # Bias term (replaces custom zeros([F_]))
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.F_,),
                initializer=zero_init,
                dtype=tf.float32,
                name='bias',
                trainable=True
            )
        else:
            self.bias = None

        # Mark layer as built (required for Keras)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass (TF2 call method replaces __call__ in TF1)"""
        # Unpack inputs: (node_features, adjacency_matrix)
        X, A = inputs  # X: (batch_size, node_num, input_dim); A: (batch_size, node_num, node_num)

        # Input dropout (only apply during training)
        if self.in_drop > 0.0 and training:
            X = tf.nn.dropout(X, rate=self.in_drop)

        outputs = []    # Store output of each attention head
        dense_masks = []# Store attention score masks (for debugging/analysis)

        for head in range(self.attn_heads):
            # Step 1: Linear transformation of node features
            kernel = self.weights_list[head]
            # TF2 tf.matmul replaces TF1 tf.tensordot (equivalent for 3D inputs)
            features = tf.matmul(X, kernel)  # (batch_size, node_num, F_)

            # Step 2: Compute attention scores
            attn_self_kernel = self.attn_self_weights[head]
            attn_neigh_kernel = self.attn_neigh_weights[head]
            attn_for_self = tf.matmul(features, attn_self_kernel)  # (batch_size, node_num, 1)
            attn_for_neigh = tf.matmul(features, attn_neigh_kernel)# (batch_size, node_num, 1)

            # Step 3: Attention head calculation (a(Wh_i, Wh_j))
            # Transpose to enable broadcasting: (batch_size, 1, node_num)
            dense = attn_for_self + tf.transpose(attn_for_neigh, [0, 2, 1])  # (batch_size, node_num, node_num)
            dense = tf.nn.leaky_relu(dense, alpha=0.2)  # Apply non-linearity

            # Step 4: Mask non-edges (replace TF1 placeholder self.A with input A)
            zero_vec = -9e15 * tf.ones_like(dense)
            dense = tf.where(A > 0.0, dense, zero_vec)
            dense_masks.append(dense)

            # Step 5: Softmax to get attention coefficients
            dense = tf.nn.softmax(dense)  # (batch_size, node_num, node_num)

            # Step 6: Dropout on attention coefficients and features (training only)
            if training and self.dropout_rate > 0.0:
                dropout_attn = tf.nn.dropout(dense, rate=self.dropout_rate)
                dropout_feat = tf.nn.dropout(features, rate=self.dropout_rate)
            else:
                dropout_attn = dense
                dropout_feat = features

            # Step 7: Aggregate neighbor features
            node_features = tf.matmul(dropout_attn, dropout_feat)  # (batch_size, node_num, F_)

            # Step 8: Add bias (if enabled)
            if self.use_bias:
                node_features += self.bias

            # Step 9: Store head output
            if self.attn_heads_reduction == 'concat':
                outputs.append(self.activation(node_features))
            else:
                outputs.append(node_features)

        # Step 10: Aggregate attention heads
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (batch_size, node_num, F_ * attn_heads)
        else:
            output = tf.add_n(outputs) / self.attn_heads  # Average heads
            output = self.activation(output)

        return output, dense_masks

# ------------------------------------------------------
# Step 5: Convert TF1 fc_layer to TF2 Keras Layer subclass
# ------------------------------------------------------
class FCLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, bias=False, name=''):
        super().__init__(name=f'fc_layer{name}')  # Call parent Layer __init__

        # Store hyperparameters
        self.input_dim = input_dim    # Input dimension
        self.output_dim = output_dim  # Output dimension
        self.dropout = dropout        # Dropout rate
        self.act = act                # Activation function
        self.bias = bias              # Whether to use bias

    def build(self, input_shape):
        """Build trainable weights (TF2 built-in initializers)"""
        glorot_init = initializers.GlorotUniform(seed=seed)
        zero_init = initializers.Zeros()

        # Linear weight (replaces custom glorot([input_dim, output_dim]))
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=glorot_init,
            dtype=tf.float32,
            name='kernel',
            trainable=True
        )

        # Bias term (replaces custom zeros([output_dim]))
        if self.bias:
            self.bias_term = self.add_weight(
                shape=(self.output_dim,),
                initializer=zero_init,
                dtype=tf.float32,
                name='bias',
                trainable=True
            )
        else:
            self.bias_term = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass for fully connected layer"""
        x = inputs

        # Dropout (training only)
        if self.dropout > 0.0 and training:
            x = tf.nn.dropout(x, rate=self.dropout)

        # Linear transformation (replace TF1 tf.tensordot)
        output = tf.matmul(x, self.kernel)

        # Add bias (if enabled)
        if self.bias:
            output += self.bias_term

        # Apply activation
        return self.act(output)

# ------------------------------------------------------
# Step 6: Convert TF1 Model to TF2 Keras Model subclass
# ------------------------------------------------------
class GATMILModel(Model):
    def __init__(self, input_dim, config):
        super().__init__(name='gat_mil')  # Call parent Model __init__

        # Store config and input dimension
        self.config = config
        self.input_dim = input_dim  # Input feature dimension (from data)

        # Initialize layers (replaces TF1 _build method)
        self._build_layers()

        # Initialize mask M (replaces TF1 tens function)
        # Use add_weight to register as trainable variable
        self.M = self.add_weight(
            shape=(self.config['node_num'], self.config['node_num']),
            initializer=initializers.Constant(value=10.0),  # TF1: tf.constant(10, ...)
            dtype=tf.float32,
            name='mask',
            trainable=True
        )

        # Initialize optimizers (TF2 replaces tf.train.AdamOptimizer)
        self.optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.optimizer_mask = optimizers.Adam(learning_rate=self.config['learning_rate_mask'])

        # Metrics (for tracking during training)
        self.train_loss_metric = metrics.Mean(name='train_loss')
        self.train_acc_metric = metrics.Mean(name='train_acc')
        self.val_loss_metric = metrics.Mean(name='val_loss')
        self.val_acc_metric = metrics.Mean(name='val_acc')

    def _build_layers(self):
        """Initialize GAT and FC layers (matches TF1 _build)"""
        # GAT Layer 1: concat heads
        self.gat1 = GATLayer(
            input_dim=self.input_dim,
            F_=self.config['hidden1_gat'],
            attn_heads=self.config['attn_heads'],
            attn_heads_reduction='concat',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            dropout_rate=self.config['dropout'],
            in_drop=self.config['in_drop'],
            name='1'
        )

        # GAT Layer 2: average heads (input_dim = hidden1_gat * attn_heads)
        self.gat2 = GATLayer(
            input_dim=self.config['hidden1_gat'] * self.config['attn_heads'],
            F_=self.config['output_gat'],
            attn_heads=3,
            attn_heads_reduction='average',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            dropout_rate=self.config['dropout'],
            in_drop=self.config['in_drop'],
            name='2'
        )

        # FC Layer 1: node-level probability (sigmoid)
        self.fc1 = FCLayer(
            input_dim=self.config['output_gat'],
            output_dim=self.config['output_dim'],
            dropout=self.config['dropout'],
            act=tf.nn.sigmoid,
            bias=False,
            name='1'
        )

        # FC Layer 2: MIL attention (softmax)
        self.fc2 = FCLayer(
            input_dim=self.config['node_num'],
            output_dim=self.config['node_num'],
            dropout=self.config['dropout'],
            act=tf.nn.softmax,
            bias=False,
            name='2'
        )

        # Store layers for easy access
        self.gat_layers = [self.gat1, self.gat2]
        self.fc_layers = [self.fc1, self.fc2]

    def call(self, inputs, training=None):
        """Forward pass (matches TF1 build method)"""
        # Unpack inputs: (node_features, adjacency_matrix)
        X, A = inputs

        # Step 1: Apply mask M (sigmoid to get [0,1] values)
        sigmoid_M = tf.sigmoid(self.M)
        # Broadcast mask to batch dimension: (batch_size, node_num, node_num)
        X = tf.multiply(X, sigmoid_M[tf.newaxis, ...])

        # Step 2: Pass through GAT layers
        gcn_activations = [X]  # Track activations (matches TF1 self.gcn_activations)
        dense_masks = []        # Track attention masks

        for layer in self.gat_layers:
            hidden, dense_mask = layer([gcn_activations[-1], A], training=training)
            gcn_activations.append(hidden)
            dense_masks.append(dense_mask)

        # Step 3: Pass through FC layers (MIL logic)
        # Node-level probability
        node_prob = self.fc1(gcn_activations[-1], training=training)  # (batch_size, node_num, 1)
        # Reshape for MIL attention: (batch_size, node_num)
        tensor = tf.reshape(node_prob, shape=(-1, self.config['node_num']))
        # MIL attention weights
        attention_prob = self.fc2(tensor, training=training)  # (batch_size, node_num)

        # Step 4: MIL pooling (weighted sum)
        attention_mul = tf.multiply(tensor, attention_prob)  # (batch_size, node_num)
        outputs = tf.reduce_sum(attention_mul, axis=1, keepdims=True)  # (batch_size, 1)

        # Store masks for analysis (optional)
        self.dense_masks = dense_masks
        return outputs

    def compute_loss(self, y_true, y_pred):
        """Compute total loss (matches TF1 _loss method)"""
        # 1. L2 weight decay (only on GAT1 weights, same as TF1)
        l2_loss = 0.0
        for weight in self.gat1.trainable_weights:
            l2_loss += self.config['weight_decay'] * tf.nn.l2_loss(weight)

        # 2. Log loss (matches TF1 tf.losses.log_loss)
        # TF2: binary_crossentropy with from_logits=False (since y_pred is sigmoid output)
        log_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

        # Total loss
        total_loss = log_loss + l2_loss
        return total_loss, log_loss, l2_loss

    def train_step(self, data, training_mask=False):
        """Single training step (uses TF2 GradientTape)"""
        X_batch, A_batch, y_batch = data

        # Use GradientTape to track gradients
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self([X_batch, A_batch], training=True)
            # Compute loss
            total_loss, _, _ = self.compute_loss(y_batch, y_pred)
            # Compute accuracy
            acc = accuracy(y_pred, y_batch)

        # Determine which variables to update
        if training_mask:
            # Only update mask M (mask optimization phase)
            trainable_vars = [self.M]
        else:
            # Update all variables except M (model training phase)
            trainable_vars = [var for var in self.trainable_variables if var.name != 'mask:0']

        # Compute gradients and apply updates
        gradients = tape.gradient(total_loss, trainable_vars)
        if training_mask:
            self.optimizer_mask.apply_gradients(zip(gradients, trainable_vars))
        else:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.train_loss_metric.update_state(total_loss)
        self.train_acc_metric.update_state(acc)

        return {'loss': self.train_loss_metric.result(), 'acc': self.train_acc_metric.result()}

    def test_step(self, data):
        """Single validation/test step"""
        X_batch, A_batch, y_batch = data

        # Forward pass (training=False disables dropout)
        y_pred = self([X_batch, A_batch], training=False)
        # Compute loss and accuracy
        total_loss, _, _ = self.compute_loss(y_batch, y_pred)
        acc = accuracy(y_pred, y_batch)

        # Update metrics
        self.val_loss_metric.update_state(total_loss)
        self.val_acc_metric.update_state(acc)

        return {'loss': self.val_loss_metric.result(), 'acc': self.val_acc_metric.result()}

    def reset_metrics(self):
        """Reset training/validation metrics"""
        self.train_loss_metric.reset_state()
        self.train_acc_metric.reset_state()
        self.val_loss_metric.reset_state()
        self.val_acc_metric.reset_state()

# ------------------------------------------------------
# Step 7: Data loading functions (keep same logic, TF2-compatible)
# ------------------------------------------------------
def load_connectivity(subject_list, kind, atlas_name='ho', data_folder='/home/celery/Data/ABIDE/ABIDE/Outputs/cpac/filt_global/mat'):#Local: /Users/celery/Research/dataset/ABIDE/Outputs/cpac/filt_global/mat
    """Load connectivity matrices (same as original)"""
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, f"{subject}_{atlas_name}_{kind}.mat")
        matrix = sio.loadmat(fl)['connectivity']
        # Remove 83rd node (atlas-specific)
        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)
        all_networks.append(matrix)
    return np.array(all_networks, dtype=np.float32)  # Use float32 for TF2

def getconn_vector(subject_name0, kind, atlas, label_dict):
    """Get features (connectivity) and labels (same as original)"""
    subject_name = np.array(subject_name0)
    # Load connectivity features
    conn_array = load_connectivity(subject_name, kind, atlas)
    data_x = np.array(conn_array, dtype=np.float32)
    # Load labels
    data_y = np.array([[int(label_dict[subname])] for subname in subject_name], dtype=np.float32)
    return data_x, data_y

def shuffle(adjs, features, y):
    """Shuffle data (same as original)"""
    shuffle_ix = np.random.permutation(np.arange(len(y)))
    return adjs[shuffle_ix], features[shuffle_ix], y[shuffle_ix]

def create_batches(X, A, y, batch_size):
    """Create batches for training (replaces original batch loop)"""
    num_samples = len(X)
    batches = []
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batches.append((X[i:end], A[i:end], y[i:end]))
    return batches

# ------------------------------------------------------
# Step 8: Main training pipeline (TF2 custom loop)
# ------------------------------------------------------
def main():
    # --------------------------
    # 8.1 Load data
    # --------------------------
    # Subject IDs and labels
    subject_IDs = np.genfromtxt('/home/celery/Project/GAT-Li-Revisited/IDs/valid_subject_ids.txt', dtype=str).tolist() #Local: /Users/celery/Research/GAT-Li-Revisited/IDs/valid_subject_ids.txt
    label_dict = Reader.get_label(subject_IDs)  # Assume Reader is compatible with TF2
    label_list = np.array([int(label_dict[x]) for x in subject_IDs])

    # Load features (connectivity) and adjacency (absolute connectivity)
    X, Y = getconn_vector(subject_IDs, "correlation", "ho", label_dict)
    adjs = np.abs(X)  # Adjacency matrix = absolute correlation

    # --------------------------
    # 8.2 K-fold setup (only run target fold)
    # --------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = [] #store results from all folds
    target_fold = 0
    # Get train/test indices for target fold (config['fold'] = 4)
    for fold_idx, (train_index, test_index) in enumerate(skf.split(subject_IDs, label_list)):

        if fold_idx != target_fold:
            print(f"Skipping Fold {fold_idx}")
            continue
        print(f"\n=== Processing Fold {fold_idx} ===")

        # Split data into train/test
        features_train, features_test = X[train_index], X[test_index]
        support_train, support_test = adjs[train_index], adjs[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Shuffle training data
        support_train, features_train, y_train = shuffle(support_train, features_train, y_train)
        support_test, features_test, y_test = shuffle(support_test, features_test, y_test)

        print(f"Fold {fold_idx} - Train shape: {features_train.shape}, Test shape: {features_test.shape}")
        print(f"Y test sample: {y_test[:3]}")

        # --------------------------
        # 8.3 Initialize model
        # --------------------------
        # Input dimension = number of features per node (last dim of features)
        input_dim = features_train.shape[-1]
        model = GATMILModel(input_dim=input_dim, config=config)

        # Model save paths (matches original)
        # New paths (with .weights.h5 extension)
        bestModelSavePath0 = '/home/celery/Project/GAT-Li-Revisited/Model_save/fold_e_mask%s/gat_e%s_weights_best.weights.h5' % (str(fold_idx), str(fold_idx)) #Users/celery/Research/GAT-Li-Revisited/Model_save
        bestModelSavePath1 = '/home/celery/Project/GAT-Li-Revisited/Model_save/fold_e_mask%s/gat_e%s_weights_best.weights.h5' % (str(fold_idx), str(fold_idx))
        # Create directory if not exists
        os.makedirs(os.path.dirname(bestModelSavePath0), exist_ok=True)

        # --------------------------
        # 8.4 Phase 1: Train model (exclude mask M)
        # --------------------------
        print("\n=== Phase 1: Train Model (Exclude Mask) ===")
        batch_size = config['batch_num']
        train_batches = create_batches(features_train, support_train, y_train, batch_size)
        val_batches = create_batches(features_test, support_test, y_test, batch_size)

        cost_val = []  # Track validation loss for early stopping
        best_val_loss = float('inf')

        for epoch in range(config['epochs']):
            # Reset metrics
            model.reset_metrics()
            start_time = time.time()

            # Train on all batches
            for batch in train_batches:
                model.train_step(batch, training_mask=False)

            # Validate on all batches
            for batch in val_batches:
                model.test_step(batch)

            # Get metrics
            train_loss = model.train_loss_metric.result()
            train_acc = model.train_acc_metric.result()
            val_loss = model.val_loss_metric.result()
            val_acc = model.val_acc_metric.result()
            cost_val.append(val_loss.numpy())

            # Print progress
            print(f"Epoch: {epoch+1:04d} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | "
                  f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f} | Time: {time.time()-start_time:.5f}")

            # Early stopping (matches original logic)
            if epoch > config['early_stopping']:
                recent_val_loss = cost_val[-(config['early_stopping']+1):-1]
                if val_loss > np.mean(recent_val_loss):
                    print("Early stopping triggered!")
                    break

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(bestModelSavePath0)  # TF2 model.save_weights

        # Load best model weights
        model.load_weights(bestModelSavePath0)
        print("\nOptimization Finished!")

        # --------------------------
        # 8.5 Evaluate Phase 1 model
        # --------------------------
        print("\n=== Evaluate Phase 1 Model ===")
        model.reset_metrics()
        for batch in val_batches:
            model.test_step(batch)
        test_loss = model.val_loss_metric.result()
        test_acc = model.val_acc_metric.result()
        print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")

        # Predict on test set
        y_pred = model([features_test, support_test], training=False).numpy()
        y_pred_class = np.round(y_pred)

        # Compute metrics (same as original)
        [[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred_class, labels=[0, 1]).astype(float)
        acc = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (FP + TN)
        sensitivity = recall = TP / (TP + FN)
        fscore = f1_score(y_test, y_pred_class)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        print(f"Accuracy: {acc:.5f} | Sensitivity: {sensitivity:.5f} | Specificity: {specificity:.5f} | "
              f"F1: {fscore:.5f} | AUC: {roc_auc:.5f}")
        result = [acc, sensitivity, specificity, fscore, roc_auc]
        l1 = [test_acc.numpy()]

        # --------------------------
        # 8.6 Phase 2: Train mask M (freeze other weights)
        # --------------------------
        print("\n=== Phase 2: Train Mask M ===")
        cost_val_mask = []
        for epoch in range(config['epochs_mask']):
            model.reset_metrics()
            start_time = time.time()

            # Train on all batches (only update M)
            for batch in train_batches:
                model.train_step(batch, training_mask=True)

            # Validate on all batches
            for batch in val_batches:
                model.test_step(batch)

            # Get metrics
            train_loss = model.train_loss_metric.result()
            train_acc = model.train_acc_metric.result()
            val_loss = model.val_loss_metric.result()
            val_acc = model.val_acc_metric.result()
            cost_val_mask.append(val_loss.numpy())

            # Print progress
            print(f"Epoch: {epoch+1:04d} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | "
                  f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f} | Time: {time.time()-start_time:.5f}")

        # Save model with optimized mask
        model.save_weights(bestModelSavePath1)
        print("\nMask Optimization Finished!")

        # --------------------------
        # 8.7 Evaluate Phase 2 (masked model)
        # --------------------------
        print("\n=== Evaluate Phase 2 (Masked Model) ===")
        model.reset_metrics()
        for batch in val_batches:
            model.test_step(batch)
        test_loss_mask = model.val_loss_metric.result()
        test_acc_mask = model.val_acc_metric.result()
        print(f"Test Loss (Masked): {test_loss_mask:.5f} | Test Acc (Masked): {test_acc_mask:.5f}")
        l2 = [test_acc_mask.numpy()]

        # --------------------------
        # 8.8 Save mask
        # --------------------------
        # Get sigmoid mask (same as original)
        s_m = tf.sigmoid(model.M).numpy()
        # Save directory
        save_dir = "/home/celery/Project/GAT-Li-Revisited/weights" #/Users/celery/Research/GAT-Li-Revisited/weights
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"fold{fold_idx}_mask.pkl")
        # Save mask
        with open(save_path, 'wb+') as f:
            pkl.dump(s_m, f)
        print(f"\nMask saved to: {save_path}")

        print(f"\nFinal Results - Phase 1 Acc: {l1} | Phase 2 Acc: {l2}")
        print(f"Full Metrics: {result}")

        all_results.append({
        'fold': fold_idx,
        'phase1_acc': test_acc.numpy(),
        'phase2_acc': test_acc_mask.numpy(),
        })

    # --------------------------
    # Print AGGREGATED results (mean accuracy across folds)
    # --------------------------
    print("\n" + "="*50)
    print("All Folds Completed! Mean Accuracy Across Folds:")
    print("="*50)

    # Calculate mean accuracy for Phase 1 and Phase 2
    mean_phase1_acc = np.mean([res['phase1_acc'] for res in all_results])
    mean_phase2_acc = np.mean([res['phase2_acc'] for res in all_results])

    print(f"Phase 1 Mean Accuracy: {mean_phase1_acc:.5f}")
    print(f"Phase 2 Mean Accuracy: {mean_phase2_acc:.5f}")



# Run main
if __name__ == "__main__":
    main()
