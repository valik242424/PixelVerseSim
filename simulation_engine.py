import random
import torch # Потрібен для роботи з тензорами next_state

from config import (GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY,
                    REWARD_FOOD, REWARD_MOVE, REWARD_WALL_COLLISION, REWARD_DEATH) # Імпортуємо винагороди
# Імпортуємо класи сутностей та функцію підготовки входу
from entities import Entity, Wall, Bot, Food, prepare_input_vector # Додали prepare_input_vector

class SimulationEngine:
    """Керує станом та логікою симуляції, тепер з підтримкою RL."""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.grid_data = [[None for _ in range(width)] for _ in range(height)]
        self.entities_list = []
        self.bots_list = []
        self.food_items = []
        self.bot_positions = {} # Словник {bot_instance: (r, c)}
        self.current_step = 0
        self._is_training = True # За замовчуванням починаємо в режимі тренування

    def initialize_world(self, num_walls=50, num_bots=10, num_food=30, bot_start_energy=(50, 150), food_energy=(15, 40)):
        """Заповнює світ початковими сутностями."""
        print("Initializing world...")
        self.grid_data = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.entities_list = []
        self.bots_list = []
        self.food_items = []
        self.bot_positions = {}
        self.current_step = 0

        def find_empty_cell():
            # ... (код find_empty_cell без змін) ...
            attempts = 0
            max_attempts = self.width * self.height
            while attempts < max_attempts:
                r = random.randint(0, self.height - 1)
                c = random.randint(0, self.width - 1)
                if self.grid_data[r][c] is None:
                    return r, c
                attempts += 1
            return None

        # 1. Стіни
        for _ in range(num_walls):
            pos = find_empty_cell()
            if pos:
                r, c = pos
                wall = Wall()
                self.grid_data[r][c] = wall
                self.entities_list.append(wall)

        # 2. Боти
        from PySide6.QtGui import QColor, Qt # Тимчасовий імпорт для кольорів
        bot_colors = [(255,0,0), (0,255,255), (255,0,255)] # Використовуємо RGB кортежі
        for i in range(num_bots):
             pos = find_empty_cell()
             if pos:
                 r, c = pos
                 start_energy = random.randint(bot_start_energy[0], bot_start_energy[1])
                 bot_color = random.choice(bot_colors)
                 # Створюємо бота (він сам створить DQNAgent)
                 bot = Bot(bot_id=f"B{i+1}", color=bot_color, energy=start_energy)
                 bot.set_training_mode(self._is_training) # Встановлюємо режим тренування
                 self.grid_data[r][c] = bot
                 self.entities_list.append(bot)
                 self.bots_list.append(bot)
                 self.bot_positions[bot] = (r, c)

        # 3. Їжа
        for _ in range(num_food):
             pos = find_empty_cell()
             if pos:
                 r, c = pos
                 energy_val = random.randint(food_energy[0], food_energy[1])
                 food = Food(energy_value=energy_val)
                 self.grid_data[r][c] = food
                 self.entities_list.append(food)
                 self.food_items.append(food)

        print(f"World initialized: {len(self.entities_list)} entities ({len(self.bots_list)} bots, {len(self.food_items)} food). Training: {self._is_training}")


    def get_grid_data(self):
        return self.grid_data

    def get_entity_at(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid_data[r][c]
        return None

    def _respawn_food(self, target_food_count=30, food_energy=(15, 40)):
        # ... (код _respawn_food без змін) ...
        current_food_count = len(self.food_items)
        needed = target_food_count - current_food_count
        if needed <= 0: return []

        spawned_logs = []
        spawn_attempts = needed * 2

        for _ in range(spawn_attempts):
            if len(self.food_items) >= target_food_count: break
            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)
            if self.grid_data[r][c] is None:
                energy_val = random.randint(food_energy[0], food_energy[1])
                new_food = Food(energy_value=energy_val)
                self.grid_data[r][c] = new_food
                self.entities_list.append(new_food)
                self.food_items.append(new_food)
        return spawned_logs

    def step(self):
        """Виконує один крок симуляції з логікою RL."""
        self.current_step += 1
        logs = []
        pending_changes = [] # (r, c, new_entity or None)
        next_bot_positions = {} # {bot_instance: (r, c)}
        consumed_food_this_step = {} # {bot: food_energy}
        wall_collisions_this_step = set() # {bot}
        dead_bots_this_step = [] # Боти, що померли на цьому кроці

        # Копіюємо список ботів, щоб ітерація була безпечною, якщо бот помре
        current_bots = list(self.bots_list)
        random.shuffle(current_bots) # Випадковий порядок обробки

        # --- Фаза 1: Оновлення ботів (рішення та потенційні дії) ---
        for bot in current_bots:
            if bot not in self.bot_positions: continue # Пропускаємо, якщо бота вже немає

            current_r, current_c = self.bot_positions[bot]
            initial_energy = bot.properties.get('energy', 0)

            # Перевірка, чи бот ще живий ПЕРЕД дією
            if initial_energy <= 0:
                # Мертвий бот не діє, його позиція не змінюється
                if bot not in next_bot_positions:
                    next_bot_positions[bot] = (current_r, current_c)
                continue # До наступного бота

            # --- Отримуємо дію від бота (він сам готує state і викликає агента) ---
            # bot.update також зменшує енергію за крок і зберігає last_state/last_action
            action_name = bot.update(self.grid_data, current_r, current_c)

            # Перевіряємо, чи бот помер ПІСЛЯ дії/витрати енергії
            final_energy_after_action = bot.properties.get('energy', 0)
            is_dead_after_action = final_energy_after_action <= 0

            if is_dead_after_action and not bot.logged_death:
                 # Це не повинно траплятися, бот сам ставить logged_death в update
                 print(f"Warning: Bot {bot.properties.get('id')} died but flag not set!")
                 bot.logged_death = True # Виправляємо

            if is_dead_after_action:
                 dead_bots_this_step.append(bot)
                 # Мертвий бот не рухається, навіть якщо вирішив рухатись
                 next_bot_positions[bot] = (current_r, current_c)
                 # Досвід буде збережено пізніше, коли розрахуємо reward/next_state
                 continue # До наступного бота

            # --- Якщо бот живий, обробляємо його дію ---
            if action_name is None: action_name = "stay" # На випадок помилки в bot.update

            intended_r, intended_c = current_r, current_c
            moved = False
            if action_name == "move_north": intended_r -= 1; moved = True
            elif action_name == "move_east": intended_c += 1; moved = True
            elif action_name == "move_south": intended_r += 1; moved = True
            elif action_name == "move_west": intended_c -= 1; moved = True

            # --- Перевірка можливості руху та взаємодії ---
            can_move = False
            ate_food = False
            hit_wall_or_border = False
            target_cell_content = None
            final_r, final_c = current_r, current_c # За замовчуванням залишається

            if moved:
                if 0 <= intended_r < self.height and 0 <= intended_c < self.width:
                    target_cell_content = self.grid_data[intended_r][intended_c]
                    target_coords = (intended_r, intended_c)

                    # Перевіряємо, чи клітинка вже не зайнята іншим ботом *на цьому кроці*
                    target_occupied_this_step = False
                    for occupied_pos in next_bot_positions.values():
                        if occupied_pos == target_coords:
                            target_occupied_this_step = True
                            break

                    if not target_occupied_this_step:
                        if isinstance(target_cell_content, Food):
                            # Перевіряємо, чи цю їжу вже не "забронював" інший бот
                            # (Проста перевірка, може бути складніша логіка)
                            is_food_claimed = False
                            for other_bot, food_energy in consumed_food_this_step.items():
                                other_bot_pos = next_bot_positions.get(other_bot)
                                if other_bot_pos == target_coords:
                                    is_food_claimed = True
                                    break
                            if not is_food_claimed:
                                can_move = True
                                ate_food = True
                        elif target_cell_content is None:
                            can_move = True
                        elif isinstance(target_cell_content, Wall):
                             hit_wall_or_border = True # Врізався в стіну
                        # else: Натрапив на іншого бота, який ще не рухався (вважаємо стіною для руху)
                        #       або на щось невідоме. Рух неможливий.
                        #       hit_wall_or_border = True # Можна вважати це зіткненням
                    else:
                        # target_occupied_this_step is True - зіткнення з іншим ботом
                        hit_wall_or_border = True # Трактуємо як зіткнення

                else: # Вихід за межі
                    hit_wall_or_border = True

            # --- Фіналізація позиції та взаємодій ---
            if can_move:
                final_r, final_c = intended_r, intended_c
                #logs.append(f"Bot {bot.properties.get('id')} moves to ({final_r},{final_c}). Action: {action_name}.")

                if ate_food and target_cell_content:
                    eaten_energy = target_cell_content.properties.get('energy', 0)
                    bot.properties['energy'] = min(MAX_ENERGY, bot.properties.get('energy', 0) + eaten_energy)
                    logs.append(f"Bot {bot.properties.get('id')} ate food at ({final_r},{final_c}). E: {bot.properties['energy']:.0f}")
                    consumed_food_this_step[bot] = eaten_energy # Записуємо, хто скільки з'їв
                    # Видалення їжі буде пізніше, при застосуванні змін

                # Запланувати зміни на сітці
                pending_changes.append((final_r, final_c, bot)) # Поставити бота сюди
                pending_changes.append((current_r, current_c, None)) # Звільнити старе місце
            else:
                # Рух не відбувся (стіна, межа, інший бот)
                final_r, final_c = current_r, current_c # Залишається на місці
                if moved and hit_wall_or_border: # Якщо намагався рухатись і вдарився
                    wall_collisions_this_step.add(bot)
                    # logs.append(f"Bot {bot.properties.get('id')} collision/stayed at ({current_r},{current_c}). Action: {action_name}.")

            # Записуємо фінальну позицію бота (навіть якщо не рухався)
            next_bot_positions[bot] = (final_r, final_c)

        # --- Фаза 2: Застосування змін на сітці ---
        # Спочатку видаляємо з'їдену їжу з сітки та списків
        food_to_remove_coords = set()
        for bot, food_energy in consumed_food_this_step.items():
            food_coord = next_bot_positions.get(bot) # Позиція бота = позиція з'їденої їжі
            if food_coord:
                food_to_remove_coords.add(food_coord)

        temp_entities_list = []
        temp_food_items = []
        for entity in self.entities_list:
            is_food = isinstance(entity, Food)
            should_keep = True
            if is_food:
                 # Знаходимо позицію їжі (це неефективно, краще мати кеш позицій їжі)
                 food_pos = None
                 for r in range(self.height):
                     for c in range(self.width):
                         if self.grid_data[r][c] == entity:
                             food_pos = (r, c)
                             break
                     if food_pos: break
                 if food_pos in food_to_remove_coords:
                     should_keep = False
                     if self.grid_data[food_pos[0]][food_pos[1]] == entity:
                         self.grid_data[food_pos[0]][food_pos[1]] = None # Видаляємо з сітки
            if should_keep:
                temp_entities_list.append(entity)
                if is_food:
                    temp_food_items.append(entity)
        self.entities_list = temp_entities_list
        self.food_items = temp_food_items


        # Потім очищаємо старі позиції ботів, які рухались
        moved_bots_changes = [change for change in pending_changes if change[2] is None]
        for r, c, _ in moved_bots_changes:
            if 0 <= r < self.height and 0 <= c < self.width:
                 # Перевіряємо, чи там часом не з'явилась їжа (не повинно, але про всяк випадок)
                 if isinstance(self.grid_data[r][c], Bot): # Тільки якщо там був бот
                     self.grid_data[r][c] = None

        # Потім ставимо ботів на нові позиції
        placed_bots_changes = [change for change in pending_changes if isinstance(change[2], Bot)]
        for r, c, bot_entity in placed_bots_changes:
             if 0 <= r < self.height and 0 <= c < self.width:
                 # Перевірка на колізії (не повинно бути через логіку вище, але...)
                 if self.grid_data[r][c] is not None and not isinstance(self.grid_data[r][c], Food): # Дозволяємо стати на місце їжі
                     logs.append(f"COLLISION DETECTED at ({r},{c})! Cell occupied by {type(self.grid_data[r][c])}, tried to place Bot {bot_entity.properties.get('id')}")
                 self.grid_data[r][c] = bot_entity

        # Оновлюємо кеш позицій ботів
        self.bot_positions = next_bot_positions

        # --- Фаза 3: Розрахунок винагород та збереження досвіду ---
        for bot in current_bots:
            if bot.last_state is None or bot.last_action_index is None:
                continue # Пропускаємо, якщо бот не діяв (був мертвий на початку або помилка)

            final_pos_r, final_pos_c = self.bot_positions.get(bot, (None, None))
            if final_pos_r is None: continue # На випадок, якщо бота видалили якось інакше

            # Визначаємо, чи бот помер на цьому кроці
            done = bot in dead_bots_this_step
            current_energy = bot.properties.get('energy', 0) # Енергія ПІСЛЯ всіх дій

            # Розрахунок винагороди
            reward = REWARD_MOVE # Базова вартість кроку
            if bot in consumed_food_this_step:
                reward += REWARD_FOOD # З'їв їжу
            if bot in wall_collisions_this_step:
                reward += REWARD_WALL_COLLISION # Врізався
            if done:
                reward += REWARD_DEATH # Помер

            # Розрахунок наступного стану (next_state)
            next_state_tensor = None
            if not done:
                # Готуємо спостереження з НОВОЇ позиції та НОВОЮ енергією
                next_state_tensor = prepare_input_vector(self.grid_data, final_pos_r, final_pos_c, current_energy)
                if next_state_tensor is None:
                     print(f"Error preparing next_state for bot {bot.properties.get('id')} at ({final_pos_r},{final_pos_c})")
                     # Що робити в цьому випадку? Можна пропустити збереження досвіду
                     continue

            # Зберігаємо досвід в буфері агента
            bot.store_experience(next_state_tensor, reward, done)

        # --- Фаза 4: Обробка смертей (видалення з активних списків) ---
        for dead_bot in dead_bots_this_step:
            r, c = self.bot_positions[dead_bot]
            if self.grid_data[r][c] == dead_bot:
                 self.grid_data[r][c] = None # Прибираємо тіло з сітки

            if dead_bot in self.bots_list: self.bots_list.remove(dead_bot)
            # З entities_list вже видалено раніше? Ні, видаляємо тут.
            if dead_bot in self.entities_list: self.entities_list.remove(dead_bot)
            if dead_bot in self.bot_positions: del self.bot_positions[dead_bot]
            logs.append(f"Bot {dead_bot.properties.get('id')} removed from simulation.")


        # --- Фаза 5: Спавн нової їжі ---
        spawn_logs = self._respawn_food()
        logs.extend(spawn_logs)

        # --- Фаза 6: Очищення last_state/action для живих ботів ---
        # Це потрібно, щоб уникнути збереження досвіду, якщо бот пропустить хід (напр., через помилку)
        # for bot in self.bots_list: # Тільки для живих
        #     bot.last_state = None
        #     bot.last_action_index = None
        # Подумаємо: можливо, це не потрібно, якщо store_experience викликається коректно.

        return logs

    # --- Методи керування тренуванням та моделями ---

    def set_training_mode(self, is_training: bool):
        """Встановлює режим навчання для всіх ботів."""
        self._is_training = is_training
        print(f"Setting training mode to: {is_training} for all bots.")
        for bot in self.bots_list:
            bot.set_training_mode(is_training)

    def reset_simulation(self):
        """Скидає симуляцію до початкового стану, включаючи стан агентів."""
        print("--- R E S E T T I N G   S I M U L A T I O N ---")
        # Реініціалізуємо світ (створює нових ботів з новими агентами)
        # TODO: Можливо, краще не перестворювати ботів, а скидати їх стан?
        # Поточна initialize_world перестворює все.
        self.initialize_world() # Це також встановить режим тренування з self._is_training

        # Якщо ми не перестворюємо ботів, а скидаємо стан:
        # self.current_step = 0
        # for bot in self.bots_list:
        #     bot.reset_state() # Скидаємо GRU стан, прапорці
        #     # Треба також скинути енергію та позицію бота
        #     # ... логіка розстановки ботів на початкові позиції ...
        # self.grid_data = ... # Очистити сітку, розставити стіни, їжу, ботів ...
        # print("Simulation reset. Bot states cleared.")

    def save_models(self, path_prefix="models/bot_model"):
        """Зберігає моделі всіх поточних ботів."""
        print(f"Saving models with prefix: {path_prefix}")
        import os
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True) # Створюємо директорію, якщо треба
        for bot in self.bots_list:
            bot_id = bot.properties.get('id', 'unknown')
            model_path = f"{path_prefix}_{bot_id}.pth"
            bot.save_model(model_path)

    def load_models(self, path_prefix="models/bot_model"):
        """Завантажує моделі для всіх поточних ботів (якщо файли існують)."""
        print(f"Loading models with prefix: {path_prefix}")
        loaded_count = 0
        for bot in self.bots_list:
            bot_id = bot.properties.get('id', 'unknown')
            model_path = f"{path_prefix}_{bot_id}.pth"
            import os
            if os.path.exists(model_path):
                bot.load_model(model_path)
                loaded_count += 1
            else:
                print(f"Model file not found for bot {bot_id}: {model_path}. Using initial weights.")
        print(f"Loaded {loaded_count} models.")