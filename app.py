from flask import Flask, request, jsonify, render_template, json
import numpy as np
from sklearn.datasets import make_blobs
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

kmeans = None
centroids_initialized = False
initial_plot = None

class KMeans:
    def __init__(self, data, k, initialization):
        self.data = data
        self.k = k
        self.initialization = initialization
        self.centers = np.zeros((self.k, self.data.shape[1]))
        self.assignments = [-1] * len(data)
        self.steps = []

    def initialize_centers(self):
        if self.initialization == 'random':
            self.centers = self.data[np.random.choice(len(self.data), size=self.k, replace=False)]
        elif self.initialization == 'farthest':
            self.centers = self.farthest_first_init()
        elif self.initialization == 'kmeans++':
            self.centers = self.kmeans_plus_plus_init()
        elif self.initialization == 'manual':
            pass

    def farthest_first_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            dists = np.min(np.linalg.norm(self.data[:, np.newaxis] - centers, axis=2), axis=0)
            next_center = self.data[np.argmax(dists)]
            centers.append(next_center)
        return np.array(centers)

    def kmeans_plus_plus_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            dists = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) for x in self.data])
            probs = dists / dists.sum()
            next_center = self.data[np.random.choice(len(self.data), p=probs)]
            centers.append(next_center)
        return np.array(centers)

    def assign_clusters(self):
        dists = np.linalg.norm(self.data[:, np.newaxis] - self.centers, axis=2)
        self.assignments = np.argmin(dists, axis=1)

    def update_centers(self):
        new_centers = np.zeros_like(self.centers)
        for i in range(self.k):
            assigned_points = self.data[self.assignments == i]
            if len(assigned_points) > 0:
                new_centers[i] = np.mean(assigned_points, axis=0)
        return new_centers

    def step(self):
        global centroids_initialized
        if not centroids_initialized:
            self.initialize_centers()
            centroids_initialized = True
        else:
            self.assign_clusters()
            new_centers = self.update_centers()
            if np.allclose(self.centers, new_centers):
                return False
            self.centers = new_centers
        return True

def generate_random_data(n_samples=10000):
    return np.random.randn(n_samples, 2)

def plot_clusters(data, centers, assignments):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[:, 0], y=data[:, 1],
        mode='markers',
        marker=dict(color=assignments, colorscale='Viridis', size=6),
        name='Data Points'
    ))
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(color='red', symbol='x', size=12, line=dict(width=2, color='DarkSlateGrey')),
            name='Centroids'
        ))
    fig.update_layout(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        showlegend=True,
        width=700,
        height=500
    )
    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global kmeans, centroids_initialized, initial_plot
    k = int(request.form['k'])
    initialization_method = request.form['initialization']
    centroids_initialized = False
    X = generate_random_data(n_samples=1000)
    kmeans = KMeans(X, k, initialization_method)
    fig = plot_clusters(kmeans.data, None, [-1] * len(kmeans.data))
    initial_plot = fig
    return jsonify(success=True, plot=fig)

@app.route('/step', methods=['POST'])
def step():
    global kmeans
    converged = not kmeans.step()
    fig = plot_clusters(kmeans.data, kmeans.centers, kmeans.assignments)
    return jsonify(converged=converged, plot=fig)

@app.route('/run', methods=['POST'])
def run():
    global kmeans
    converged = False
    while not converged:
        converged = not kmeans.step()
    fig = plot_clusters(kmeans.data, kmeans.centers, kmeans.assignments)
    return jsonify(converged=True, plot=fig)

@app.route('/reset', methods=['POST'])
def reset():
    global kmeans, centroids_initialized, initial_plot
    if kmeans and initial_plot:
        kmeans.assignments = [-1] * len(kmeans.data)
        kmeans.centers = np.zeros((kmeans.k, kmeans.data.shape[1]))
        centroids_initialized = False
        return jsonify(success=True, plot=initial_plot, clear_manual_centroids=True)
    else:
        return jsonify(success=False, error="Initial state not available.")

@app.route('/manual_centroids', methods=['POST'])
def manual_centroids():
    global kmeans
    centroids = json.loads(request.form['centroids'])
    manual_centers = np.array([[c['x'], c['y']] for c in centroids])
    if len(manual_centers) > kmeans.k:
        manual_centers = manual_centers[:kmeans.k]
    manual_centers[:, 0] = manual_centers[:, 0] * (kmeans.data[:, 0].max() / 700)
    manual_centers[:, 1] = manual_centers[:, 1] * (kmeans.data[:, 1].max() / 500)
    kmeans.centers = manual_centers
    kmeans.assign_clusters()
    fig = plot_clusters(kmeans.data, kmeans.centers, kmeans.assignments)
    return jsonify(plot=fig)

@app.route('/get_plot_data', methods=['GET'])
def get_plot_data():
    global kmeans
    fig = plot_clusters(kmeans.data, kmeans.centers, kmeans.assignments)
    return jsonify(pio.to_json(fig))

if __name__ == '__main__':
    app.run(debug=True)