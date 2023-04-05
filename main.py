import random
import sys

import numpy as np
import pygame
import pygame.gfxdraw as gfxdraw
import pygame.locals as pl
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional
orange = 227, 146, 59
HEIGHT = 600
WIDTH = 600
ACC = 0.5
FRIC = -0.12
FPS = 30
ZOOM_SPEED = 1.1
ORANGE = 227, 146, 59
WHITE = 255, 255, 255
BLACK = 0, 0, 0

FramePerSec = pygame.time.Clock()
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))


class Blob(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.radius = 10
        self.surf = pygame.Surface(
            (self.radius * 2 + 4, self.radius * 2 + 4), pygame.SRCALPHA
        )
        self.color = color
        self.rect = self.surf.get_rect()
        self.pos = vec((x, y))
        self.vel = vec((0, 0))
        border_color = (255, 255, 255)
        self.border_radius = self.radius + 1

    def render(self, scale, offset):
        radius = int(self.radius * scale)
        screen_pos = self.pos * scale + offset

        if (0 - radius) < screen_pos.x < (WIDTH + radius) and (
            0 - radius
        ) < screen_pos.y < (HEIGHT + radius):
            # Draw the border circle with antialiasing
            gfxdraw.aacircle(
                displaysurface, int(screen_pos.x), int(screen_pos.y), radius + 1, WHITE
            )
            gfxdraw.filled_circle(
                displaysurface, int(screen_pos.x), int(screen_pos.y), radius + 1, WHITE
            )

            # Draw the original circle on top of the border circle with antialiasing
            gfxdraw.aacircle(
                displaysurface, int(screen_pos.x), int(screen_pos.y), radius, self.color
            )
            gfxdraw.filled_circle(
                displaysurface, int(screen_pos.x), int(screen_pos.y), radius, self.color
            )

    @classmethod
    def initialize_centroids(cls, blobs, n_clusters):
        cls.centroids = [random.choice(blobs).pos for _ in range(n_clusters)]

    @classmethod
    def update_centroids(cls, blobs):
        centroid_sums = [vec(0, 0) for _ in range(len(cls.centroids))]
        centroid_counts = [0] * len(cls.centroids)

        for blob in blobs:
            closest_center_idx = min(
                range(len(cls.centroids)),
                key=lambda i: blob.pos.distance_to(cls.centroids[i]),
            )
            centroid_sums[closest_center_idx] += blob.pos
            centroid_counts[closest_center_idx] += 1

        for i, count in enumerate(centroid_counts):
            if count > 0:
                cls.centroids[i] = centroid_sums[i] / count
            else:
                cls.centroids[i] = random.choice(blobs).pos

    @classmethod
    def draw_centroids(cls, scale, offset):
        centroid_color = (255, 255, 255)  # White
        centroid_size = 14

        for center in cls.centroids:
            x, y = center
            screen_pos = vec(x, y) * scale + offset
            centroid_rect = pygame.Rect(0, 0, centroid_size, centroid_size)
            centroid_rect.center = (int(screen_pos.x), int(screen_pos.y))
            pygame.draw.rect(displaysurface, centroid_color, centroid_rect)

    def update_position(self, cluster_centers):
        closest_center = None
        closest_distance = None

        for center in cluster_centers:
            center_tuple = tuple(center)
            distance = self.pos.distance_to(vec(center_tuple))

            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_center = vec(center_tuple)

        self.pos = self.pos.lerp(closest_center, 0.01)

    def __repr__(self) -> str:
        return f"Blob({self.pos.x}, {self.pos.y}, {self.color})"


def zoom_in(scale, mouse_pos, offset):
    new_scale = scale * ZOOM_SPEED
    new_offset = offset - (mouse_pos - offset) * (ZOOM_SPEED - 1)
    return new_scale, new_offset


def zoom_out(scale, mouse_pos, offset):
    new_scale = scale / ZOOM_SPEED
    new_offset = offset + (mouse_pos - offset) * (1 - 1 / ZOOM_SPEED)
    return new_scale, new_offset


def normalize_zoom():
    return 1.0


def generate_blobs(n_samples, centers, cluster_centers):
    data = make_blobs(
        n_samples=n_samples,
        centers=centers,
        random_state=4,
        cluster_std=cluster_centers,
    )
    points = data[0]
    # Scale points to fit within the display surface
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x) * (WIDTH - 40) + 20
    points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y) * (HEIGHT - 40) + 20

    return points


def cluster_points(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    return kmeans.predict(points), kmeans.cluster_centers_


# Main function
def main():
    n_samples = 100
    centers = 6
    cluster_centers = 1.0
    points = generate_blobs(n_samples, centers, cluster_centers=cluster_centers)

    n_clusters = centers
    labels, cluster_centers = cluster_points(points, n_clusters)

    # Create blobs
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    blobs = [
        Blob(x, y, colors[labels[i] % len(colors)]) for i, (x, y) in enumerate(points)
    ]

    # Initialize zoom and pan variables
    scale = 1.0
    offset = vec(0, 0)
    dragging = False
    drag_start = vec(0, 0)

    # Main loop
    while True:
        if not hasattr(Blob, "centroids"):
            Blob.initialize_centroids(blobs, n_clusters)
        # Perform KMeans clustering
        Blob.update_centroids(blobs)
        Blob.draw_centroids(scale, offset)
        # Update blob positions based on centroids
        for blob in blobs:
            blob.update_position(cluster_centers)

        mouse_pos = vec(pygame.mouse.get_pos())

        for event in pygame.event.get():
            if event.type == pl.QUIT:
                pygame.quit()
                sys.exit()

            # Mouse wheel event for zooming
            if event.type == pl.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    scale, offset = zoom_in(scale, mouse_pos, offset)
                elif event.button == 5:  # Scroll down
                    scale, offset = zoom_out(scale, mouse_pos, offset)
                elif event.button == 2:  # Middle click
                    scale = normalize_zoom()
                    offset = vec(0, 0)
                elif event.button == 3:  # Right click
                    dragging = True
                    drag_start = mouse_pos

            if (
                event.type == pl.MOUSEBUTTONUP and event.button == 3
            ):  # Right click release
                dragging = False

        # Update the offset while dragging
        if dragging:
            offset += mouse_pos - drag_start
            drag_start = mouse_pos

        displaysurface.fill(BLACK)

        # Render blobs with the current scale and offset
        for blob in blobs:
            blob.render(scale, offset)

        Blob.draw_centroids(scale, offset)

        pygame.display.update()
        FramePerSec.tick(FPS)


if __name__ == "__main__":
    main()
