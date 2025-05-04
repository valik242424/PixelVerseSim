GRID_WIDTH = 100
GRID_HEIGHT = 100
VIEW_SIZE = 20 # Розмір видимого вікна (для рендерингу)
CELL_SIZE_PX = 30
MAX_ENERGY = 200
ENERGY_COST_STEP = 1.0

# --- Параметри поля зору та входу для RL ---
VIEW_RADIUS = 2            # Радіус огляду бота (квадрат 5x5)
CELL_TYPE_EMPTY = 0
CELL_TYPE_WALL = 1
CELL_TYPE_BOT = 2
CELL_TYPE_FOOD = 3
NUM_CELL_TYPES = 4         # Кількість типів клітинок для one-hot encoding
FIELD_SIZE = (2 * VIEW_RADIUS + 1) # Розмір сторони квадрата огляду (5)
# Розмір вхідного вектора: (площа_огляду * кількість_типів) + 1 (енергія)
INPUT_SIZE = (FIELD_SIZE ** 2) * NUM_CELL_TYPES + 1 # (5*5)*4 + 1 = 101

# --- DQN Гіперпараметри ---
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LEARNING_RATE = 1e-4
TARGET_UPDATE_INTERVAL = 10
LEARN_START_SIZE = 1000
LEARN_EVERY_N_STEPS = 4

# --- Винагороди ---
REWARD_FOOD = 25.0
REWARD_MOVE = -0.1
REWARD_WALL_COLLISION = -1.0
REWARD_DEATH = -50.0