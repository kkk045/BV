# BV
import folium
import math
import requests
import webbrowser
import numpy as np
import csv
import os
import random
import time
import pickle
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic


class RouteOptimizer:
    def __init__(self, cities: Dict[str, Tuple[float, float]], final_point: Tuple[float, float]):
        self.cities = cities
        self.city_names = list(cities.keys())
        self.city_coords = np.array(list(cities.values()))
        self.final_point = final_point
        self.route_cache = {}
        self.ml_model = None
        self.load_ml_model()
        self.fuel_price = 55.0  # руб/км (примерная стоимость топлива)
        self.driver_cost = 2000  # руб/день (стоимость водителя)
        self.avg_speed = 60  # км/ч (средняя скорость)

    def load_ml_model(self):
        """Загрузка или создание модели машинного обучения"""
        model_file = 'route_decision_model.pkl'

        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.ml_model = pickle.load(f)
        else:
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            X, y = self.generate_training_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.ml_model.fit(X_train, y_train)
            preds = self.ml_model.predict(X_test)
            print(f"Model accuracy: {accuracy_score(y_test, preds):.2f}")
            with open(model_file, 'wb') as f:
                pickle.dump(self.ml_model, f)

    def generate_training_data(self):
        """Генерация синтетических данных для обучения модели"""
        X = []
        y = []

        for _ in range(1000):
            num_cities = random.randint(3, 15)
            cities_coords = np.random.rand(num_cities, 2) * 10 + [45, 45]
            features = self.calculate_features(cities_coords)
            max_distance = np.max([geodesic(c1, c2).km for c1 in cities_coords for c2 in cities_coords])
            label = 1 if max_distance > 500 else 0
            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)

    def calculate_features(self, coords):
        """Вычисление признаков для модели ML"""
        distances = [geodesic(c1, c2).km for c1 in coords for c2 in coords]
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        num_cities = len(coords)
        min_lat, max_lat = min(c[0] for c in coords), max(c[0] for c in coords)
        min_lon, max_lon = min(c[1] for c in coords), max(c[1] for c in coords)
        area = (max_lat - min_lat) * (max_lon - min_lon)
        density = num_cities / (area + 1e-6)
        return [mean_distance, max_distance, std_distance, num_cities, density]

    def should_use_hub(self):
        """Определение, нужно ли использовать хаб (на основе ML)"""
        features = self.calculate_features(self.city_coords)
        return self.ml_model.predict([features])[0] == 1

    def get_route_distance(self, route_indices: List[int]) -> float:
        """Вычисление длины маршрута с кэшированием результатов"""
        route_key = tuple(route_indices)
        if route_key in self.route_cache:
            return self.route_cache[route_key]

        route_coords = [self.city_coords[i] for i in route_indices] + [self.final_point]

        if len(route_coords) == 2:
            return geodesic(route_coords[0], route_coords[1]).km

        coords_str = ";".join([f"{lon},{lat}" for lat, lon in route_coords])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=false"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok':
                    distance = data['routes'][0]['distance'] / 1000  # км
                    self.route_cache[route_key] = distance
                    return distance
        except:
            pass
        return float('inf')

    def find_central_hub(self):
        """Находит наиболее центральный город для сбора людей"""
        min_total_distance = float('inf')
        best_hub = None

        for i, (city, coords) in enumerate(self.cities.items()):
            total_distance = 0
            for j, (other_city, other_coords) in enumerate(self.cities.items()):
                if i != j:
                    total_distance += geodesic(coords, other_coords).km

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_hub = (city, coords)

        return best_hub

    def optimize_direct_route(self):
        """Оптимизация прямого маршрута через все города"""
        city_indices = list(range(len(self.city_names)))
        best_order, distance = self.optimize_route_order(city_indices)
        return best_order, distance

    def optimize_hub_route(self):
        """Оптимизация маршрута через центральный хаб"""
        hub_city, hub_coords = self.find_central_hub()
        hub_index = list(self.cities.keys()).index(hub_city)

        to_hub_distances = []
        for i, (city, coords) in enumerate(self.cities.items()):
            if city != hub_city:
                dist = self.get_route_distance([i, hub_index])
                to_hub_distances.append(dist)

        hub_to_end = self.get_route_distance([hub_index])
        total_distance = sum(to_hub_distances) + hub_to_end
        route_order = [i for i in range(len(self.city_names)) if i != hub_index] + [hub_index]

        return route_order, total_distance, hub_city

    def optimize_route_order(self, indices):
        """Оптимизация порядка посещения городов"""
        if len(indices) <= 1:
            return indices, 0

        population = [random.sample(indices, len(indices)) for _ in range(50)]
        best_order = None
        best_distance = float('inf')

        for _ in range(50):
            fitness = []
            for ind in population:
                distance = self.get_route_distance(ind)
                fitness.append((ind, distance))
                if distance < best_distance:
                    best_distance = distance
                    best_order = ind.copy()

            fitness.sort(key=lambda x: x[1])
            next_generation = [x[0] for x in fitness[:10]]

            while len(next_generation) < 50:
                parent1 = random.choice(fitness[:25])[0]
                parent2 = random.choice(fitness[:25])[0]
                child = self.ordered_crossover(parent1, parent2)

                if random.random() < 0.1:
                    child = self.mutate(child)

                next_generation.append(child)

            population = next_generation

        return best_order, best_distance

    def ordered_crossover(self, parent1, parent2):
        """Упорядоченный кроссовер (OX)"""
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))

        child = [-1] * size
        child[a:b] = parent1[a:b]

        ptr = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[ptr] in child[a:b]:
                    ptr += 1
                child[i] = parent2[ptr]
                ptr += 1

        return child

    def mutate(self, individual):
        """Мутация - перестановка двух городов"""
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    def calculate_route_cost(self, distance):
        """Расчет стоимости маршрута"""
        hours = distance / self.avg_speed
        days = max(1, round(hours / 8))  # 8-часовой рабочий день
        fuel_cost = distance * self.fuel_price
        driver_cost = days * self.driver_cost
        return fuel_cost + driver_cost

    def compare_routes(self, direct_distance, hub_distance):
        """Сравнение маршрутов и объяснение выбора"""
        direct_cost = self.calculate_route_cost(direct_distance)
        hub_cost = self.calculate_route_cost(hub_distance)

        features = self.calculate_features(self.city_coords)
        mean_dist = features[0]
        max_dist = features[1]
        density = features[4]

        reasons = []

        if hub_distance < direct_distance:
            reasons.append(f"Маршрут через хаб короче на {direct_distance - hub_distance:.1f} км")
            reasons.append(f"Экономия: {direct_cost - hub_cost:.2f} руб")
        else:
            reasons.append(f"Прямой маршрут короче на {hub_distance - direct_distance:.1f} км")
            reasons.append(f"Экономия: {hub_cost - direct_cost:.2f} руб")

        if mean_dist > 300:
            reasons.append("Среднее расстояние между городами большое (>300 км)")
        if max_dist > 500:
            reasons.append("Есть города на большом расстоянии друг от друга (>500 км)")
        if density < 0.01:
            reasons.append("Низкая плотность городов (большие расстояния между ними)")

        return reasons

    def optimize_route(self):
        """Автоматическая оптимизация маршрута с детальным отчетом"""
        if len(self.cities) <= 3:
            print("\nМало городов (≤3) - используем прямой маршрут")
            best_order, distance = self.optimize_direct_route()

            # Рассчитываем альтернативный вариант через хаб для сравнения
            _, hub_distance, _ = self.optimize_hub_route()
            reasons = ["Малое количество городов делает прямой маршрут оптимальным"]

            return {
                'route': best_order,
                'distance': distance,
                'hub_city': None,
                'route_type': 'direct',
                'reasons': reasons,
                'comparison': {
                    'direct_distance': distance,
                    'hub_distance': hub_distance,
                    'direct_cost': self.calculate_route_cost(distance),
                    'hub_cost': self.calculate_route_cost(hub_distance)
                }
            }

        use_hub = self.should_use_hub()

        if use_hub:
            print("\nИспользуем маршрут через центральный хаб")
            best_order, distance, hub_city = self.optimize_hub_route()

            # Рассчитываем альтернативный прямой маршрут для сравнения
            _, direct_distance = self.optimize_direct_route()
            reasons = self.compare_routes(direct_distance, distance)

            return {
                'route': best_order,
                'distance': distance,
                'hub_city': hub_city,
                'route_type': 'hub',
                'reasons': reasons,
                'comparison': {
                    'direct_distance': direct_distance,
                    'hub_distance': distance,
                    'direct_cost': self.calculate_route_cost(direct_distance),
                    'hub_cost': self.calculate_route_cost(distance)
                }
            }
        else:
            print("\nИспользуем прямой маршрут")
            best_order, distance = self.optimize_direct_route()

            # Рассчитываем альтернативный вариант через хаб для сравнения
            _, hub_distance, _ = self.optimize_hub_route()
            reasons = self.compare_routes(distance, hub_distance)

            return {
                'route': best_order,
                'distance': distance,
                'hub_city': None,
                'route_type': 'direct',
                'reasons': reasons,
                'comparison': {
                    'direct_distance': distance,
                    'hub_distance': hub_distance,
                    'direct_cost': self.calculate_route_cost(distance),
                    'hub_cost': self.calculate_route_cost(hub_distance)
                }
            }


def load_cities_from_csv(file_path):
    """Загрузка данных из CSV файла"""
    cities = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")

    encodings = ['utf-8-sig', 'windows-1251', 'cp1251', 'iso-8859-5']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.DictReader(file, delimiter=';')

                if not all(col in reader.fieldnames for col in ['город', 'широта', 'долгота']):
                    raise ValueError(f"Не найдены нужные колонки. Найдены: {reader.fieldnames}")

                for row in reader:
                    try:
                        city = row['город'].strip().replace('’', "'").replace('—', '-')
                        lat = row['широта'].replace(',', '.').strip()
                        lon = row['долгота'].replace(',', '.').strip()

                        if not city or not lat or not lon:
                            continue

                        cities[city] = (float(lat), float(lon))
                    except ValueError as e:
                        print(f"Ошибка в строке: {row}. Ошибка: {e}")

            print(f"Успешно загружено {len(cities)} городов")
            return cities

        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Ошибка при чтении в кодировке {encoding}: {str(e)}")

    raise ValueError("Не удалось прочитать файл. Проверьте кодировку и формат.")


def select_cities(all_cities):
    """Интерактивный выбор городов"""
    available_cities = sorted(all_cities.keys())

    print("\nДоступные города:")
    for i, city in enumerate(available_cities, 1):
        print(f"{i}. {city}")

    while True:
        choice = input("\nВведите номера городов через запятую (или 'all' для всех): ").strip()

        if choice.lower() == 'all':
            return all_cities

        try:
            indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            invalid = [idx for idx in indices if idx < 0 or idx >= len(available_cities)]

            if invalid:
                print(f"Ошибка: неверные номера {invalid}. Допустимый диапазон: 1-{len(available_cities)}")
                continue

            return {available_cities[idx]: all_cities[available_cities[idx]] for idx in indices}

        except ValueError:
            print("Ошибка: вводите только номера через запятую")


def select_end_point(available_cities, exclude_cities):
    """Выбор конечной точки (исключая выбранные города отправления)"""
    possible_ends = [city for city in available_cities if city not in exclude_cities]

    print("\nДоступные города назначения:")
    for i, city in enumerate(possible_ends, 1):
        print(f"{i}. {city}")

    while True:
        try:
            end_choice = int(input("\nВведите номер города назначения: ").strip())
            if 1 <= end_choice <= len(possible_ends):
                return possible_ends[end_choice - 1]
            print(f"Ошибка: введите число от 1 до {len(possible_ends)}")
        except ValueError:
            print("Ошибка: введите число")


def get_route(coordinates):
    """Получение маршрута через OSRM API"""
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Ошибка при запросе маршрута: {response.status_code}")

    data = response.json()
    if data['code'] != 'Ok':
        raise Exception(f"Ошибка в данных маршрута: {data['message']}")

    return data['routes'][0]['geometry']['coordinates']


def create_route_map(cities, final_point, best_route=None, distance=None, hub_city=None):
    """Создание карты с маршрутом и инфраструктурой"""
    city_names = list(cities.keys())
    city_coords = list(cities.values())

    lats = [coord[0] for coord in city_coords] + [final_point[0]]
    lons = [coord[1] for coord in city_coords] + [final_point[1]]
    map_center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(location=map_center, zoom_start=7, tiles='CartoDB positron')

    INFRASTRUCTURE_DATA = {
        'default': {
            'Транспорт': ['Автовокзал', 'ЖД станция'],
            'Медицина': ['Больница', 'Поликлиника'],
            'Торговля': ['ТЦ', 'Рынок']
        },
        'Нижневартовск': {
            'Транспорт': ['Аэропорт', 'Речной порт'],
            'Медицина': ['ГКБ', 'Детская больница'],
            'Культура': ['Драмтеатр', 'Музей']
        }
    }

    for i, (city, coords) in enumerate(cities.items()):
        infra = INFRASTRUCTURE_DATA.get(city, INFRASTRUCTURE_DATA['default'])
        popup_content = f"<b>{city}</b><hr>"

        if hub_city and city == hub_city:
            popup_content = f"<b>ХАБ: {city}</b><hr>"

        for category, items in infra.items():
            popup_content += f"<b>{category}:</b><br>"
            popup_content += "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

        if hub_city and city == hub_city:
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='purple', icon='star')
            ).add_to(m)
        else:
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue')
            ).add_to(m)

    if best_route and distance:
        route_points = [city_coords[i] for i in best_route] + [final_point]

        try:
            full_route = get_route(route_points)
            route_color = 'purple' if hub_city else 'red'

            folium.PolyLine(
                [(p[1], p[0]) for p in full_route],
                color=route_color,
                weight=5,
                opacity=0.8,
                popup=f"Маршрут ({distance:.1f} км)"
            ).add_to(m)

            for i in range(0, len(full_route) - 1, max(1, len(full_route) // 10)):
                p1, p2 = full_route[i], full_route[i + 1]
                angle = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1])) + 90

                folium.Marker(
                    location=(p2[1], p2[0]),
                    icon=folium.DivIcon(
                        icon_size=(10, 10),
                        icon_anchor=(5, 5),
                        html=f'<div style="font-size: 10px; color: {route_color}; transform: rotate({angle}deg);">▶</div>'
                    )
                ).add_to(m)
        except Exception as e:
            print(f"Ошибка при построении маршрута: {e}")

    folium.Marker(
        location=final_point,
        popup=f"<b>Конечная точка</b><br>{final_point}",
        icon=folium.Icon(color='green', icon='flag')
    ).add_to(m)

    if best_route:
        route_info = "<div style='position:fixed;bottom:20px;left:20px;background:white;padding:10px;'>"
        route_info += f"<b>Маршрут:</b> {distance:.1f} км<ol>"

        for idx in best_route:
            city_name = city_names[idx]
            if hub_city and city_name == hub_city:
                route_info += f"<li><b>ХАБ: {city_name}</b></li>"
            else:
                route_info += f"<li>{city_name}</li>"

        route_info += "</ol></div>"
        m.get_root().html.add_child(folium.Element(route_info))

    return m


def print_route_report(result, cities, end_city):
    """Вывод детального отчета о маршруте"""
    city_names = list(cities.keys())

    print("\n" + "=" * 50)
    print("ДЕТАЛЬНЫЙ ОТЧЕТ О МАРШРУТЕ")
    print("=" * 50)

    print("\n🔹 ОПТИМАЛЬНЫЙ МАРШРУТ:")
    if result['route_type'] == 'hub':
        print(f"Тип: через центральный хаб ({result['hub_city']})")
    else:
        print("Тип: прямой маршрут")

    print("\n🔹 ПОСЛЕДОВАТЕЛЬНОСТЬ ГОРОДОВ:")
    for i, idx in enumerate(result['route'], 1):
        print(f"{i}. {city_names[idx]}" + (
            " (ХАБ)" if result['hub_city'] and city_names[idx] == result['hub_city'] else ""))
    print(f"Конечная точка: {end_city}")

    print("\n🔹 ПАРАМЕТРЫ МАРШРУТА:")
    print(f"Общее расстояние: {result['distance']:.1f} км")
    cost = result['comparison'][f"{result['route_type']}_cost"]
    print(f"Примерная стоимость: {cost:.2f} руб (топливо + водитель)")

    print("\n🔹 СРАВНЕНИЕ С АЛЬТЕРНАТИВНЫМИ МАРШРУТАМИ:")
    print(
        f"Прямой маршрут: {result['comparison']['direct_distance']:.1f} км ({result['comparison']['direct_cost']:.2f} руб)")
    print(f"Через хаб: {result['comparison']['hub_distance']:.1f} км ({result['comparison']['hub_cost']:.2f} руб)")

    print("\n🔹 ПРИЧИНЫ ВЫБОРА ЭТОГО МАРШРУТА:")
    for reason in result['reasons']:
        print(f"- {reason}")

    print("\n" + "=" * 50)

def main():
    print("Программа оптимизации маршрутов для сбора людей")

    try:
        all_cities = load_cities_from_csv('cities.csv')
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Проверьте наличие и формат файла cities.csv")
        return

    selected_cities = select_cities(all_cities)
    print("\nВыбраны города для маршрутизации:", ", ".join(selected_cities.keys()))

    end_city = select_end_point(selected_cities.keys(), [])
    final_point = all_cities[end_city]
    print(f"\nТочка назначения: {end_city}")

    optimizer = RouteOptimizer(selected_cities, final_point)
    result = optimizer.optimize_route()

    print_route_report(result, selected_cities, end_city)

    output_file = "optimized_route_map.html"
    m = create_route_map(
        selected_cities,
        final_point,
        best_route=result['route'],
        distance=result['distance'],
        hub_city=result['hub_city']
    )

    folium.Marker(
        location=final_point,
        popup=f"<b>Точка назначения: {end_city}</b>",
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    m.save(output_file)
    print(f"\nКарта маршрута сохранена в файл {output_file}")
    webbrowser.open(output_file)

if __name__ == "__main__":
    main()
