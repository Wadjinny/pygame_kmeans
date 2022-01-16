import pygame
from pygame.locals import *
import sys
from sklearn.datasets._samples_generator import make_blobs
pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional
orange = 227, 146, 59
HEIGHT = 450
WIDTH = 400
ACC = 0.5
FRIC = -0.12
FPS = 60
 
FramePerSec = pygame.time.Clock()
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))

class Blob(pygame.sprite.Sprite):
    def __init__(self,x,y):
        super().__init__() 
        self.radius = 10
        self.surf = pygame.Surface((self.radius*2, self.radius*2))
        self.color = orange
        self.rect = self.surf.get_rect()
        self.pos = vec((self.radius, self.radius))
        self.vel = vec((0,0))
    def move(self):
        pressed_keys = pygame.key.get_pressed()  
        self.vel = vec((0,0))    

        if pressed_keys[K_LEFT]:
            self.vel.x = -1
        if pressed_keys[K_RIGHT]:
            self.vel.x = 1
        if pressed_keys[K_DOWN]:
            self.vel.y = 1
        if pressed_keys[K_UP]:
            self.vel.y = -1
            
        
        self.pos += self.vel
        self.rect.center = self.pos  
    def render(self):
        pygame.draw.circle(self.surf,self.color,(self.radius,self.radius),self.radius)
        displaysurface.blit(self.surf, self.rect)

pygame.display.set_caption("Game")
blob = Blob()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
     
    displaysurface.fill((0,0,0))
    blob.move()
    blob.render()
    pygame.display.update()
    FramePerSec.tick(FPS)

