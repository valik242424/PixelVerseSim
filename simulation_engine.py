# refactored_code/simulation_engine.py
import random
from config import GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY
# Імпортуємо класи сутностей з нашого рефакторнутого файлу
from entities import Entity, Wall, Bot, Food

class SimulationEngine:
    """Керує станом та логікою симуляції."""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.grid_data = [[None for _ in range(width)] for _ in range(height)]
        self.entities_list = [] # Список всіх сутностей (для оновлення)
        self.bots_list = []     # Окремий список тільки ботів (для зручності)
        self.food_items = []    # Окремий список їжі
        self.bot_positions = {} # Словник {bot_instance: (r, c)}
        self.current_step = 0   # Лічильник кроків симуляції

    def initialize_world(self, num_walls=50, num_bots=10, num_food=30, bot_start_energy=(50, 150), food_energy=(15, 40)):
        """Заповнює світ початковими сутностями."""
        print("Initializing world...")
        self.grid_data = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.entities_list = []
        self.bots_list = []
        self.food_items = []
        self.bot_positions = {}
        self.current_step = 0

        # Функція для пошуку вільного місця
        def find_empty_cell():
            attempts = 0
            max_attempts = self.width * self.height # Щоб уникнути нескінченного циклу
            while attempts < max_attempts:
                r = random.randint(0, self.height - 1)
                c = random.randint(0, self.width - 1)
                if self.grid_data[r][c] is None:
                    return r, c
                attempts += 1
            return None # Не вдалося знайти вільне місце

        # 1. Стіни
        for _ in range(num_walls):
            pos = find_empty_cell()
            if pos:
                r, c = pos
                wall = Wall()
                self.grid_data[r][c] = wall
                self.entities_list.append(wall) # Стіни теж сутності, хоч і пасивні

        # 2. Боти
        from PySide6.QtGui import QColor, Qt # Тимчасовий імпорт для кольорів
        bot_colors = [QColor(Qt.GlobalColor.red), QColor(Qt.GlobalColor.cyan), QColor(Qt.GlobalColor.magenta)]
        for i in range(num_bots):
             pos = find_empty_cell()
             if pos:
                 r, c = pos
                 start_energy = random.randint(bot_start_energy[0], bot_start_energy[1])
                 bot_color = random.choice(bot_colors)
                 bot = Bot(bot_id=f"B{i+1}", color=bot_color, energy=start_energy)
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

        print(f"World initialized: {len(self.entities_list)} entities ({len(self.bots_list)} bots, {len(self.food_items)} food).")


    def get_grid_data(self):
        """Повертає поточний стан сітки."""
        return self.grid_data

    def get_entity_at(self, r, c):
        """Повертає сутність за вказаними координатами або None."""
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid_data[r][c]
        return None

    def _respawn_food(self, target_food_count=30, food_energy=(15, 40)):
        """Додає їжу, якщо її стало менше за цільову кількість."""
        current_food_count = len(self.food_items)
        needed = target_food_count - current_food_count
        if needed <= 0:
            return [] # Нічого не робимо

        spawned_logs = []
        spawn_attempts = needed * 2 # Даємо трохи більше спроб

        for _ in range(spawn_attempts):
            if len(self.food_items) >= target_food_count: break # Досягли мети

            r = random.randint(0, self.height - 1)
            c = random.randint(0, self.width - 1)

            if self.grid_data[r][c] is None: # Тільки в порожню клітинку
                energy_val = random.randint(food_energy[0], food_energy[1])
                new_food = Food(energy_value=energy_val)
                self.grid_data[r][c] = new_food
                self.entities_list.append(new_food)
                self.food_items.append(new_food)
                # spawned_logs.append(f"Spawned food at ({r},{c}) energy={energy_val}")

        return spawned_logs


    def step(self):
        """Виконує один крок симуляції."""
        self.current_step += 1
        logs = []
        pending_changes = [] # (r, c, new_entity or None)
        next_bot_positions = {} # {bot_instance: (r, c)}
        consumed_food_coords = set() # Координати з'їденої їжі (r, c)
        dead_bots_this_step = [] # Боти, що померли на цьому кроці

        # Перемішуємо список ботів для випадкового порядку обробки
        random.shuffle(self.bots_list)

        # --- Фаза 1: Оновлення ботів (рішення та потенційні дії) ---
        for bot in self.bots_list:
            if bot not in self.bot_positions: continue # На випадок, якщо бота якось видалили раніше

            current_r, current_c = self.bot_positions[bot]

            # Перевірка, чи бот ще живий ПЕРЕД дією
            current_energy = bot.properties.get('energy', 0)
            if current_energy <= 0:
                # Мертвий бот не діє, але його позиція має бути в next_bot_positions
                if bot not in next_bot_positions:
                    next_bot_positions[bot] = (current_r, current_c)
                continue # До наступного бота

            # Отримуємо дію від мозку (цей метод також зменшує енергію)
            action_name = bot.update(self.grid_data, current_r, current_c)

            # Перевіряємо, чи бот помер ПІСЛЯ дії/витрати енергії
            new_energy = bot.properties.get('energy', 0)
            if new_energy <= 0 and not bot.logged_death:
                 # bot.update вже встановив logged_death=True
                 logs.append(f"Bot {bot.properties.get('id')} ran out of energy at ({current_r},{current_c})!")
                 dead_bots_this_step.append(bot)
                 # Мертвий бот не рухається, навіть якщо вирішив рухатись
                 next_bot_positions[bot] = (current_r, current_c)
                 continue # До наступного бота

            # Якщо бот живий, обробляємо його дію
            if action_name is None: action_name = "stay"

            new_r, new_c = current_r, current_c
            moved = False
            if action_name == "move_north": new_r -= 1; moved = True
            elif action_name == "move_east": new_c += 1; moved = True
            elif action_name == "move_south": new_r += 1; moved = True
            elif action_name == "move_west": new_c -= 1; moved = True

            # --- Перевірка можливості руху та взаємодії ---
            can_move = False
            ate_food = False
            target_cell_content = None

            if moved:
                if 0 <= new_r < self.height and 0 <= new_c < self.width:
                    target_cell_content = self.grid_data[new_r][new_c]
                    target_coords = (new_r, new_c)

                    if isinstance(target_cell_content, Food) and target_coords not in consumed_food_coords:
                        can_move = True
                        ate_food = True
                    elif target_cell_content is None:
                        can_move = True
                    # else: Стіна або інший бот (статично) - рух неможливий
                # else: Вихід за межі - рух неможливий

            # --- Вирішення конфліктів та фіналізація позиції ---
            final_r, final_c = current_r, current_c # За замовчуванням залишається

            if can_move:
                target_occupied_this_step = False
                for occupied_pos in next_bot_positions.values():
                    if occupied_pos == (new_r, new_c):
                        target_occupied_this_step = True
                        #logs.append(f"Bot {bot.properties.get('id')} wanted ({new_r},{new_c}), but it was claimed.")
                        break

                if not target_occupied_this_step:
                    # Успішний рух без конфлікту
                    final_r, final_c = new_r, new_c
                    #logs.append(f"Bot {bot.properties.get('id')} moves to ({final_r},{final_c}). Action: {action_name}.")

                    # Обробка поїдання їжі
                    if ate_food and target_cell_content:
                        eaten_energy = target_cell_content.properties.get('energy', 0)
                        bot.properties['energy'] = min(MAX_ENERGY, bot.properties.get('energy', 0) + eaten_energy)
                        logs.append(f"Bot {bot.properties.get('id')} ate food at ({new_r},{new_c}). E: {bot.properties['energy']:.0f}")
                        consumed_food_coords.add(target_coords)
                        # Видалення їжі зі списків (з grid_data вона видалиться при перезаписі ботом)
                        if target_cell_content in self.food_items: self.food_items.remove(target_cell_content)
                        if target_cell_content in self.entities_list: self.entities_list.remove(target_cell_content)

                    # Запланувати зміни на сітці
                    pending_changes.append((final_r, final_c, bot)) # Поставити бота сюди
                    pending_changes.append((current_r, current_c, None)) # Звільнити старе місце
                # else: Конфлікт, бот залишається на місці (final_r, final_c не змінилися)
            # else: Рух неможливий (стіна/межа), бот залишається на місці

            # Записуємо фінальну позицію бота (навіть якщо не рухався)
            next_bot_positions[bot] = (final_r, final_c)


        # --- Фаза 2: Застосування змін на сітці ---
        # Спочатку очищаємо старі позиції ботів, які рухались
        moved_bots_changes = [change for change in pending_changes if change[2] is None]
        for r, c, _ in moved_bots_changes:
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid_data[r][c] = None

        # Потім ставимо ботів на нові позиції
        placed_bots_changes = [change for change in pending_changes if isinstance(change[2], Bot)]
        for r, c, bot_entity in placed_bots_changes:
             if 0 <= r < self.height and 0 <= c < self.width:
                 # Перевірка на всяк випадок, чи клітинка вже не зайнята (не повинно бути)
                 if self.grid_data[r][c] is not None and self.grid_data[r][c] != bot_entity:
                     logs.append(f"COLLISION DETECTED at ({r},{c})! Cell occupied by {type(self.grid_data[r][c])}, tried to place Bot {bot_entity.properties.get('id')}")
                 self.grid_data[r][c] = bot_entity

        # Оновлюємо кеш позицій ботів
        self.bot_positions = next_bot_positions

        # --- Фаза 3: Обробка смертей (видалення з активних списків) ---
        # Можна замінити мертвих ботів чимось (напр., "органікою") або просто видалити
        for dead_bot in dead_bots_this_step:
            r, c = self.bot_positions[dead_bot]
            if self.grid_data[r][c] == dead_bot: # Перевірка, чи він ще там
                 self.grid_data[r][c] = None # Прибираємо тіло з сітки
                 # Можна створити щось на його місці:
                 # self.grid_data[r][c] = OrganicMatter(value=10)
                 # self.entities_list.append(self.grid_data[r][c])

            if dead_bot in self.bots_list: self.bots_list.remove(dead_bot)
            if dead_bot in self.entities_list: self.entities_list.remove(dead_bot)
            if dead_bot in self.bot_positions: del self.bot_positions[dead_bot]
            logs.append(f"Bot {dead_bot.properties.get('id')} removed from simulation.")


        # --- Фаза 4: Спавн нової їжі ---
        spawn_logs = self._respawn_food()
        logs.extend(spawn_logs)


        return logs # Повертаємо логи для відображення в UI