import asyncio
import EEGUtils
import keras
import pygame
import sys
import tensorflow as tf

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_RADIUS = 7
FPS = 60


class Pong:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Pong Game')

        self.left_paddle = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = pygame.Rect(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

        self.ball_speed_x = 7 * (-1 if pygame.time.get_ticks() % 2 == 0 else 1)
        self.ball_speed_y = 7 * (-1 if pygame.time.get_ticks() % 2 == 0 else 1)

        self.paddle_speed = 10

        self.left_score = 0
        self.right_score = 0

        self.font = pygame.font.Font(None, 74)

        self.clock = pygame.time.Clock()

        self.model = keras.models.load_model("model.keras")

        self.predictor = EEGUtils.TestClassifier(self.model, 1297, 2, ema=.99)
        asyncio.run(self.predictor.initialize_loop())

    def draw_objects(self):
        self.screen.fill(BLACK)

        pygame.draw.rect(self.screen, WHITE, self.left_paddle)
        pygame.draw.rect(self.screen, WHITE, self.right_paddle)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)

        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

        left_text = self.font.render(str(self.left_score), True, WHITE)
        right_text = self.font.render(str(self.right_score), True, WHITE)
        self.screen.blit(left_text, (WIDTH // 4, 20))
        self.screen.blit(right_text, (WIDTH - WIDTH // 4, 20))

    def move_ball(self):
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_speed_y *= -1

        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            self.ball_speed_x *= -1

        if self.ball.left <= 0:
            self.right_score += 1
            self.reset_ball()

        if self.ball.right >= WIDTH:
            self.left_score += 1
            self.reset_ball()

    def reset_ball(self):
        self.ball.center = (WIDTH // 2, HEIGHT // 2)
        self.ball_speed_x *= -1
        self.ball_speed_y *= -1

    def handle_paddle_movement(self, predictions):
        print(predictions)
        if predictions[0][0] >= predictions[0][1]:
            predictions = 0
        else:
            predictions = 1
        print(predictions)
        keys = pygame.key.get_pressed()

        if predictions == 1 and self.left_paddle.top > 0:
            self.left_paddle.y -= self.paddle_speed
        if predictions == 0 and self.left_paddle.bottom < HEIGHT:
            self.left_paddle.y += self.paddle_speed

        if keys[pygame.K_UP] and self.right_paddle.top > 0:
            self.right_paddle.y -= self.paddle_speed
        if keys[pygame.K_DOWN] and self.right_paddle.bottom < HEIGHT:
            self.right_paddle.y += self.paddle_speed

    def run(self):
        while True:
            asyncio.run(self.predictor.main_loop())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.handle_paddle_movement(self.predictor.last_predicted)

            self.move_ball()

            self.draw_objects()

            pygame.display.flip()

            self.clock.tick(FPS)


if __name__ == '__main__':
    Pong().run()
