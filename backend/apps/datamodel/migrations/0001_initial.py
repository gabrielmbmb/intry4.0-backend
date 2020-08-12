# Generated by Django 3.0.8 on 2020-08-12 16:28

import django.contrib.postgres.fields
import django.contrib.postgres.fields.jsonb
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import re
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='Model name', max_length=128)),
                ('is_training', models.BooleanField(default=False, help_text='Wether the model is being trained or not')),
                ('trained', models.BooleanField(default=False, help_text='Wether the model is trained or not')),
                ('deployed', models.BooleanField(default=False, help_text='Wether the model is deployed or not')),
                ('date_trained', models.DateTimeField(blank=True, default=None, help_text='Date the model was trained', null=True)),
                ('date_deployed', models.DateTimeField(blank=True, default=None, help_text='Date the model was deployed', null=True)),
                ('num_predictions', models.IntegerField(default=0, help_text='Number of predictions made by this model')),
                ('task_status', models.CharField(blank=True, help_text='URL to get the progress of training process', max_length=512, null=True)),
                ('plcs', django.contrib.postgres.fields.jsonb.JSONField()),
                ('contamination', models.FloatField(blank=True, default=0.1, help_text='Contamination fraction in the training dataset', null=True, validators=[django.core.validators.MinValueValidator(0.0)])),
                ('scaler', models.CharField(blank=True, default='minmax', help_text='The scaler used to scale the data before training and predicting', max_length=48, null=True)),
                ('pca_mahalanobis', models.BooleanField(blank=True, default=False, null=True)),
                ('n_components', models.IntegerField(blank=True, default=2, help_text='Numbers of components for the PCA algorithm', null=True, validators=[django.core.validators.MinValueValidator(1)])),
                ('autoencoder', models.BooleanField(blank=True, default=False, null=True)),
                ('hidden_neurons', models.CharField(blank=True, default='32,16,16,32', help_text='Neural Network layers and the number of neurons in each layer', max_length=128, null=True, validators=[django.core.validators.RegexValidator(re.compile('^\\d+(?:,\\d+)*\\Z'), code='invalid', message='It should be a string with a list of integers separeted by a comma')])),
                ('dropout_rate', models.FloatField(blank=True, default=0.2, help_text='Dropout rate across all the layers of the Neural Network', null=True)),
                ('activation', models.CharField(blank=True, choices=[('elu', 'elu'), ('softmax', 'softmax'), ('selu', 'selu'), ('softplus', 'softplus'), ('softsign', 'softsign'), ('relu', 'relu'), ('tanh', 'tanh'), ('sigmoid', 'sigmoid'), ('hard_sigmoid', 'hard_sigmoid'), ('exponential', 'exponential')], default='elu', help_text='Layers activation function of Neural Network', max_length=24, null=True)),
                ('kernel_initializer', models.CharField(blank=True, choices=[('Zeros', 'Zeros'), ('Ones', 'Ones'), ('Constant', 'Constant'), ('RandomNormal', 'RandomNormal'), ('RandomUniform', 'RandomUniform'), ('TruncatedNormal', 'TruncatedNormal'), ('VarianceScaling', 'VarianceScaling'), ('Orthogonal', 'Orthogonal'), ('Identity', 'Identity'), ('lecun_uniform', 'lecun_uniform'), ('glorot_normal', 'glorot_normal'), ('glorot_uniform', 'glorot_uniform'), ('he_normal', 'he_normal'), ('lecun_normal', 'lecun_normal'), ('he_uniform', 'he_uniform')], default='glorot_uniform', help_text='Layers kernel initializer of Neural Network', max_length=24, null=True)),
                ('loss_function', models.CharField(blank=True, default='mse', help_text='Loss function of the Neural Network', max_length=24, null=True)),
                ('optimizer', models.CharField(blank=True, choices=[('sgd', 'sgd'), ('rmsprop', 'rmsprop'), ('adagrad', 'adagrad'), ('adadelta', 'adadelta'), ('adam', 'adam'), ('adamax', 'adamax'), ('nadam', 'nadam')], default='adam', help_text='Optimizer of Neural Network', max_length=24, null=True)),
                ('epochs', models.IntegerField(blank=True, default=100, help_text='Number of times that all the batches will be processed in the  Neural Network', null=True)),
                ('batch_size', models.IntegerField(blank=True, default=32, help_text='Batch size', null=True)),
                ('validation_split', models.FloatField(blank=True, default=0.05, help_text='Percentage of the training data that will be used for purpouses in the Neural Network', null=True)),
                ('early_stopping', models.BooleanField(blank=True, default=False, help_text="Stops the training process in the Neural Network when it's not getting any improvement", null=True)),
                ('kmeans', models.BooleanField(blank=True, default=False, null=True)),
                ('n_clusters', models.IntegerField(blank=True, default=None, help_text='Number of clusters for the K-Means algorithm', null=True)),
                ('max_cluster_elbow', models.IntegerField(blank=True, default=100, help_text='Maximun number of cluster to test in the Elbow Method', null=True)),
                ('ocsvm', models.BooleanField(blank=True, default=False, null=True)),
                ('kernel', models.CharField(blank=True, choices=[('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'), ('precomputed', 'precomputed')], default='rbf', help_text='Kernel type for One Class SVM', max_length=24, null=True)),
                ('degree', models.IntegerField(blank=True, default=3, help_text='Degree of the polynomal kernel function for One Class SVM', null=True)),
                ('gamma', models.CharField(blank=True, default='scale', help_text="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' in One Class SVM. It can 'scale', 'auto' or float", max_length=24, null=True)),
                ('coef0', models.FloatField(blank=True, default=0.0, help_text="Independent term in kernel function for One Class SVM. Only significant in 'poly'", null=True)),
                ('tol', models.FloatField(blank=True, default=0.001, help_text='Tolerance for stopping criterion for One Class SVM', null=True)),
                ('shrinking', models.BooleanField(blank=True, default=True, help_text='Whether to use the shrinking heuristic for One Class SVM', null=True)),
                ('cache_size', models.IntegerField(blank=True, default=200, help_text='Specify the size of the kernel cache in MB for One Class SVM', null=True)),
                ('gaussian_distribution', models.BooleanField(blank=True, default=False, null=True)),
                ('epsilon_candidates', models.IntegerField(blank=True, default=100000000, help_text='Number of epsilon values that will be tested to find the best one', null=True)),
                ('isolation_forest', models.BooleanField(blank=True, default=False, null=True)),
                ('n_estimators', models.IntegerField(blank=True, default=100, help_text='The number of base estimators in the ensemble for Isolation Forest', null=True)),
                ('max_features', models.FloatField(blank=True, default=1.0, help_text='Number of features to draw from X to train each base estimator for Isolation Forest', null=True)),
                ('bootstrap', models.BooleanField(blank=True, default=False, help_text='Indicates if the Bootstrap technique is going to be applied for Isolation FOrest', null=True)),
                ('lof', models.BooleanField(blank=True, default=False, null=True)),
                ('n_neighbors_lof', models.IntegerField(blank=True, default=20, help_text='Number of neighbors to use in LOF', null=True)),
                ('algorithm_lof', models.CharField(blank=True, choices=[('ball_tree', 'ball_tree'), ('kd_tree', 'kd_tree'), ('brute', 'brute'), ('auto', 'auto')], default='auto', help_text='Algorithm used to compute the nearest neighbors in LOF', max_length=24, null=True)),
                ('leaf_size_lof', models.IntegerField(blank=True, default=30, help_text='Leaf size passed to BallTree or KDTree in LOF', null=True)),
                ('metric_lof', models.CharField(blank=True, default='minkowski', help_text='The distance metric to use for the tree in LOF', max_length=24, null=True)),
                ('p_lof', models.IntegerField(blank=True, default=2, help_text='Paremeter of the Minkowski metric in LOF', null=True)),
                ('knn', models.BooleanField(blank=True, default=False, null=True)),
                ('n_neighbors_knn', models.IntegerField(blank=True, default=5, help_text='Number of neighbors to use in KNN', null=True)),
                ('radius', models.FloatField(blank=True, default=1.0, help_text='The range of parameter space to use by default for radius_neighbors', null=True)),
                ('algorithm_knn', models.CharField(blank=True, choices=[('ball_tree', 'ball_tree'), ('kd_tree', 'kd_tree'), ('brute', 'brute'), ('auto', 'auto')], default='auto', help_text='Algorithm used to compute the nearest neighbors in KNN', max_length=24, null=True)),
                ('leaf_size_knn', models.IntegerField(blank=True, default=30, help_text='Leaf size passed to BallTree or KDTree in KNN', null=True)),
                ('metric_knn', models.CharField(blank=True, default='minkowski', help_text='The distance metric to use for the tree in KNN', max_length=24, null=True)),
                ('p_knn', models.IntegerField(blank=True, default=2, help_text='Paremeter of the Minkowski metric in knn', null=True)),
                ('score_func', models.CharField(blank=True, choices=[('max_distance', 'max_distance'), ('average', 'average'), ('median', 'median')], default='max_distance', help_text='The function used to score anomalies in KNN', max_length=24, null=True)),
                ('subscriptions', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=128), default=list, size=None)),
                ('data_from_subscriptions', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
                ('dates', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
            ],
        ),
        migrations.CreateModel(
            name='TrainFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(help_text='A CSV training file containing the columns of the DataModel', upload_to='')),
                ('index_column', models.CharField(blank=True, max_length=128, null=True)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('datamodel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='datamodel.DataModel')),
            ],
            options={
                'get_latest_by': 'uploaded_at',
            },
        ),
        migrations.CreateModel(
            name='DatamodelPrediction',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('data', django.contrib.postgres.fields.jsonb.JSONField()),
                ('dates', django.contrib.postgres.fields.jsonb.JSONField()),
                ('predictions', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
                ('task_status', models.CharField(blank=True, help_text='URL to get the progress of predicting process', max_length=512, null=True)),
                ('ack', models.BooleanField(default=False)),
                ('user_ack', models.CharField(blank=True, max_length=128, null=True)),
                ('datamodel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='datamodel.DataModel')),
            ],
        ),
    ]
