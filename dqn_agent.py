import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# Імпортуємо нашу нейронку та конфіг
from bot_brain import BotBrainNet, ACTIONS, ACTION_MAP, GRU_HIDDEN_SIZE
from config import (BUFFER_SIZE, BATCH_SIZE, GAMMA, EPS_START, EPS_END,
                    EPS_DECAY, TAU, LEARNING_RATE, TARGET_UPDATE_INTERVAL,
                    LEARN_START_SIZE, LEARN_EVERY_N_STEPS, INPUT_SIZE)

# Визначаємо структуру для збереження переходів у буфері
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Визначаємо, чи доступний GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    """Фіксованого розміру буфер для зберігання переходів досвіду."""
    def __init__(self, capacity=BUFFER_SIZE):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Зберігає перехід."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=BATCH_SIZE):
        """Вибирає випадковий батч переходів."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Повертає поточний розмір буфера."""
        return len(self.memory)

class DQNAgent:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        # Використовуємо INPUT_SIZE з config
        self.input_size = INPUT_SIZE
        self.output_size = len(ACTIONS)
        self.gru_hidden_size = GRU_HIDDEN_SIZE

        # Створюємо мережі, використовуючи self.input_size
        self.policy_net = BotBrainNet(self.input_size, self.output_size, self.gru_hidden_size).to(device)
        self.target_net = BotBrainNet(self.input_size, self.output_size, self.gru_hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.policy_hidden_state = torch.zeros(1, 1, self.gru_hidden_size, device=device)
        self.is_training = True

    def select_action(self, state_tensor):
        """
        Вибирає дію за допомогою epsilon-greedy стратегії.
        state_tensor - це вже готовий тензор вхідних даних для мережі.
        """
        sample = random.random()
        # Розраховуємо поточне значення epsilon
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold or not self.is_training:
            # Жадібна дія: вибираємо найкращу дію за передбаченням мережі
            with torch.no_grad():
                # Переконуємося, що state_tensor на правильному девайсі
                state_tensor = state_tensor.to(device)
                # Отримуємо Q-значення та новий прихований стан від policy_net
                # Передаємо поточний прихований стан агента
                action_scores, next_hidden = self.policy_net(state_tensor, self.policy_hidden_state)
                # Оновлюємо прихований стан агента для наступного кроку
                self.policy_hidden_state = next_hidden.detach()
                # Вибираємо дію з максимальним Q-значенням
                action_index = action_scores.argmax().view(1, 1) # .view(1,1) для сумісності з buffer
                return action_index
        else:
            # Випадкова дія
            # Важливо: Якщо ми робимо випадкову дію, чи треба оновлювати прихований стан GRU?
            # Так, бо мережа все одно має "бачити" цей крок, навіть якщо дія випадкова.
            # Треба прогнати стан через мережу, щоб отримати next_hidden, але ігнорувати action_scores.
            with torch.no_grad():
                 state_tensor = state_tensor.to(device)
                 _, next_hidden = self.policy_net(state_tensor, self.policy_hidden_state)
                 self.policy_hidden_state = next_hidden.detach()

            action_index = torch.tensor([[random.randrange(self.output_size)]], device=device, dtype=torch.long)
            return action_index

    def store_transition(self, state, action, next_state, reward, done):
        """Зберігає перехід у буфері."""
        # Переконуємося, що всі тензори на CPU перед збереженням (буфер зазвичай на CPU)
        state = state.cpu() if state is not None else None
        action = action.cpu() if action is not None else None
        next_state = next_state.cpu() if next_state is not None else None
        reward = torch.tensor([reward], dtype=torch.float32).cpu()
        done = torch.tensor([done], dtype=torch.bool).cpu()

        # Важливо: Ми не зберігаємо прихований стан GRU в буфері.
        # DQN зазвичай працює з незалежними станами. Якщо потрібна пам'ять між батчами,
        # треба використовувати складніші підходи (напр., DRQN - Deep Recurrent Q-Network),
        # де в буфер зберігаються цілі послідовності.
        # Для простоти поки що будемо вважати кожен стан незалежним при навчанні з буфера.
        # Це означає, що GRU буде ефективно працювати тільки при виборі дії,
        # а при навчанні його "пам'ять" буде обмежена одним кроком.
        # Це спрощення, але для початку може спрацювати.
        self.memory.push(state, action, next_state, reward, done)

        # Запускаємо навчання, якщо умови виконані
        if self.is_training and len(self.memory) > LEARN_START_SIZE and self.steps_done % LEARN_EVERY_N_STEPS == 0:
            self._learn()

        # Оновлюємо цільову мережу, якщо час
        if self.is_training and self.steps_done % TARGET_UPDATE_INTERVAL == 0:
            self._update_target_network()

    def _learn(self):
        """Виконує один крок навчання DQN."""
        if len(self.memory) < BATCH_SIZE:
            return # Недостатньо даних у буфері

        # Вибираємо батч з буфера
        transitions = self.memory.sample(BATCH_SIZE)
        # Транспонуємо батч (див. документацію PyTorch DQN tutorial для пояснення)
        batch = Transition(*zip(*transitions))

        # Створюємо маску для не-фінальних наступних станів
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        # Збираємо тензори для не-фінальних наступних станів
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        # Збираємо тензори станів, дій та винагород
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # --- Розрахунок Q(s_t, a) ---
        # Мережа policy_net обчислює Q(s_t, a) для всіх дій,
        # а потім ми вибираємо ті, що відповідають діям, які були реально зроблені (action_batch).
        # Важливо: При навчанні з буфера ми не використовуємо збережений self.policy_hidden_state,
        # а ініціалізуємо його нулями для кожного елемента батчу.
        # Це спрощення, яке ігнорує довгострокову пам'ять GRU під час навчання.
        batch_size_actual = state_batch.size(0)
        initial_hidden_state_batch = torch.zeros(1, batch_size_actual, self.gru_hidden_size, device=device)

        # Проганяємо батч станів через policy_net
        # Оскільки forward очікує (batch, seq, feature), а ми маємо (batch, feature),
        # нам треба або змінити forward, або додати вимір seq=1.
        # Або, якщо forward вже обробляє один крок, передавати (batch, feature).
        # Поточний forward приймає (1, feature) і h_prev=(1, 1, H).
        # Треба адаптувати для батчу.
        # Найпростіше - змінити forward в BotBrainNet, щоб він приймав батч.
        # АБО тут зробити цикл по батчу (повільно).
        # АБО передати весь батч в GRU (batch_first=True).

        # --- Тимчасове рішення: Змінимо BotBrainNet.forward для батчу ---
        # Поки що припустимо, що BotBrainNet.forward може обробити батч state_batch
        # і повернути Q-значення для всіх дій розміром (batch_size, num_actions).
        # І що він не потребує h_prev при навчанні з буфера (або приймає None).
        # state_action_values = self.policy_net(state_batch, None)[0].gather(1, action_batch) # [0] бо forward повертає (scores, h_next)

        # --- Альтернатива: Зберігаємо поточну логіку forward, але викликаємо для батчу ---
        # Це буде повільніше, але не потребує зміни BotBrainNet зараз.
        # Збираємо Q-значення для зроблених дій
        q_values_list = []
        for i in range(batch_size_actual):
            state_i = state_batch[i].unsqueeze(0) # (feature) -> (1, feature)
            h_prev_i = torch.zeros(1, 1, self.gru_hidden_size, device=device) # Нульовий стан для навчання
            q_scores_i, _ = self.policy_net(state_i, h_prev_i) # Отримуємо (num_actions)
            q_values_list.append(q_scores_i)
        state_action_values = torch.stack(q_values_list).gather(1, action_batch)


        # --- Розрахунок V(s_{t+1}) для всіх наступних станів ---
        # Використовуємо target_net для стабільності.
        # Q-значення для наступних станів обчислюються target_net,
        # а потім вибирається max_a' Q_target(s_{t+1}, a').
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
             # Так само, як і з policy_net, треба обробити батч в target_net
             q_next_values_list = []
             initial_hidden_target = torch.zeros(1, non_final_next_states.size(0), self.gru_hidden_size, device=device)
             # Потрібно передати non_final_next_states в target_net
             # Знову ж таки, або міняти forward, або цикл/обробка батчу.
             # Використаємо цикл для послідовності:
             for i in range(non_final_next_states.size(0)):
                 state_i_next = non_final_next_states[i].unsqueeze(0)
                 h_prev_i_next = torch.zeros(1, 1, self.gru_hidden_size, device=device)
                 q_scores_i_next, _ = self.target_net(state_i_next, h_prev_i_next)
                 q_next_values_list.append(q_scores_i_next)

             # Вибираємо максимальне Q-значення для кожного наступного стану
             next_state_max_q = torch.stack(q_next_values_list).max(1)[0] # max повертає (values, indices)
             next_state_values[non_final_mask] = next_state_max_q.detach() # detach(), бо ми не хочемо градієнтів по target_net

        # --- Розрахунок очікуваних Q-значень (цільових) ---
        # Expected Q = r + gamma * V(s_{t+1})
        # Для фінальних станів V(s_{t+1}) = 0
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # --- Розрахунок функції втрат ---
        # Використовуємо Huber loss (Smooth L1 Loss) для більшої стабільності
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- Оптимізація ---
        self.optimizer.zero_grad()
        loss.backward()
        # Обмеження градієнтів (запобігає "вибуху" градієнтів)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _update_target_network(self):
        """Оновлює ваги цільової мережі."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        # print(f"[{self.bot_id}] Target network updated at step {self.steps_done}") # Для дебагу

    def save_model(self, path):
        """Зберігає ваги policy_net."""
        print(f"[{self.bot_id}] Saving model to {path}...")
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """Завантажує ваги для policy_net та копіює їх в target_net."""
        print(f"[{self.bot_id}] Loading model from {path}...")
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Оновлюємо target net
            self.policy_net.eval() # Якщо завантажили для використання, а не тренування
            self.target_net.eval()
            # Скидаємо прихований стан при завантаженні нової моделі
            self.policy_hidden_state = torch.zeros(1, 1, self.gru_hidden_size, device=device)
            print(f"[{self.bot_id}] Model loaded successfully.")
        except Exception as e:
            print(f"[{self.bot_id}] Error loading model: {e}. Using initial weights.")

    def set_training_mode(self, is_training: bool):
        """Встановлює режим навчання (впливає на epsilon-greedy та оновлення)."""
        self.is_training = is_training
        if is_training:
            self.policy_net.train() # Переводимо policy_net в режим навчання (впливає на dropout, batchnorm, якщо є)
        else:
            self.policy_net.eval() # Переводимо в режим оцінки
        print(f"[{self.bot_id}] Training mode set to: {self.is_training}")
        # Скидаємо прихований стан при зміні режиму? Можливо, варто.
        self.policy_hidden_state = torch.zeros(1, 1, self.gru_hidden_size, device=device)
