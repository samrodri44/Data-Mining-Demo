import pygame
import sys
import os
import time
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np

pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
ITEMS = [
    "Milk", "Bread", "Flour", "Egg", "Apple", "Cereal", "Banana",
    "Cheese", "Chicken", "Beef", "Rice", "Pasta", "Basketball",
    "Soccerball", "Light Bulb", "Motor Oil", "Diapers", "Baby Formula",
    "Towel", "Ice"
]
FONT = pygame.font.SysFont("Arial", 20)
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 80
BUTTON_PADDING = 20
CART_WIDTH = 300
DARK_MODE_COLORS = {
    "bg": (34, 36, 37),
    "btn": (68, 71, 72),
    "cart": (60, 63, 65),
    "text": (255, 255, 255),
    "border": (130, 130, 130)
}
LIGHT_MODE_COLORS = {
    "bg": (255, 255, 255),
    "btn": (220, 220, 220),
    "cart": (240, 240, 240),
    "text": (0, 0, 0),
    "border": (200, 200, 200)
}

ICON_FOLDER = "icons"
item_icons = {}
for item in ITEMS:
    icon_path = os.path.join(ICON_FOLDER, f"{item.lower().replace(' ', '_')}.png")
    if os.path.exists(icon_path):
        item_icons[item] = pygame.image.load(icon_path)
        item_icons[item] = pygame.transform.scale(item_icons[item], (32, 32))

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Supermarket Simulator")
clock = pygame.time.Clock()

transactions = []
current_cart = []
selected_items = set()
dark_mode = False
in_checkout = False
progress = 0
num_clusters = 3
min_support_threshold = 2

scroll_offset = 0
scroll_speed = 20


def draw_text(text, pos, color, center=False):
    txt = FONT.render(text, True, color)
    rect = txt.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    screen.blit(txt, rect)


class Button:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.hovered = False
        self.selected = False
        self.scale = 1.0
        self.prev_hovered = False

    def draw(self, colors):
        color = colors["btn"]
        border_color = colors["border"]
        text_color = colors["text"]
        base_rect = self.rect.inflate((self.scale - 1) * self.rect.width, (self.scale - 1) * self.rect.height)
        pygame.draw.rect(screen, color, base_rect, border_radius=15)
        pygame.draw.rect(screen, border_color, base_rect, width=3, border_radius=15)
        if self.selected:
            pygame.draw.rect(screen, (20, 195, 160), base_rect, width=4, border_radius=15)
        draw_text(self.text, base_rect.center, text_color, center=True)
        if self.text in item_icons:
            icon = item_icons[self.text]
            screen.blit(icon, (base_rect.left + 8, base_rect.centery - 16))

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        target_scale = 1.05 if self.hovered else 1.0
        self.scale += (target_scale - self.scale) * 0.2
        self.prev_hovered = self.hovered

    def click(self):
        self.selected = not self.selected
        return self.selected


def create_buttons():
    buttons = []
    x, y = BUTTON_PADDING, 60
    for item in ITEMS:
        if x + BUTTON_WIDTH + BUTTON_PADDING > WIDTH - CART_WIDTH:
            x = BUTTON_PADDING
            y += BUTTON_HEIGHT + BUTTON_PADDING + 10
        buttons.append(Button(item, x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
        x += BUTTON_WIDTH + BUTTON_PADDING
    return buttons

buttons = create_buttons()
checkout_btn = Button("Checkout", WIDTH - CART_WIDTH + 30, HEIGHT - 80, CART_WIDTH - 60, 50)
toggle_dark_btn = Button("", 30, HEIGHT - 80, 50, 50)
toggle_dark_btn.draw_circle = True
mining_btn = Button("Data Mining", WIDTH - CART_WIDTH + 30, HEIGHT - 140, CART_WIDTH - 60, 50)
back_btn = Button("Back", WIDTH // 2 - 50, HEIGHT - 80, 100, 40)
kmeans_btn = Button("K-Means", WIDTH // 2 - 150, HEIGHT - 140, 120, 40)
fpgrowth_btn = Button("FP-Growth", WIDTH // 2 + 30, HEIGHT - 140, 120, 40)


def animate_checkout():
    global progress, in_checkout
    in_checkout = True
    for i in range(101):
        progress = i
        draw_checkout_progress()
        pygame.display.flip()
        pygame.time.wait(10)
    in_checkout = False


def draw_checkout_progress():
    color = DARK_MODE_COLORS if dark_mode else LIGHT_MODE_COLORS
    screen.fill(color["bg"])
    draw_text("Processing Checkout...", (WIDTH // 2, HEIGHT // 2 - 40), color["text"], center=True)
    pygame.draw.rect(screen, color["btn"], (WIDTH // 2 - 150, HEIGHT // 2, 300, 30), border_radius=10)
    pygame.draw.rect(screen, (20, 195, 160), (WIDTH // 2 - 150, HEIGHT // 2, 3 * progress, 30), border_radius=10)
    draw_text(f"{progress}%", (WIDTH // 2, HEIGHT // 2 + 40), color["text"], center=True)


def run_kmeans():
    item_to_index = {item: idx for idx, item in enumerate(ITEMS)}
    X = np.zeros((len(transactions), len(ITEMS)))
    for i, trans in enumerate(transactions):
        for item in trans:
            X[i, item_to_index[item]] = 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    return kmeans.labels_


def run_fp_growth():
    freq_items = defaultdict(int)
    for trans in transactions:
        for i in range(1, len(trans) + 1):
            for combo in combinations(sorted(set(trans)), i):
                freq_items[combo] += 1
    return {k: v for k, v in freq_items.items() if v >= min_support_threshold}


def show_visualization(title, data):
    global scroll_offset
    show = True
    while show:
        screen.fill((30, 30, 30))
        draw_text(title, (WIDTH // 2, 50), (255, 255, 255), center=True)

        for i, line in enumerate(data):
            y_pos = 120 + i * 25 - scroll_offset
            if 100 <= y_pos <= HEIGHT - 100:
                draw_text(line, (100, y_pos), (255, 255, 255))

        back_btn.update(pygame.mouse.get_pos())
        back_btn.draw({"btn": (100, 100, 100), "text": (255, 255, 255), "border": (150, 150, 150)})

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if back_btn.hovered:
                    scroll_offset = 0
                    show = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    scroll_offset += scroll_speed
                elif event.key == pygame.K_UP:
                    scroll_offset = max(scroll_offset - scroll_speed, 0)
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    scroll_offset = max(scroll_offset - scroll_speed, 0)
                elif event.y < 0:
                    scroll_offset += scroll_speed
        pygame.display.flip()
        clock.tick(30)


def show_data_mining_screen():
    show = True
    while show:
        screen.fill((50, 50, 50))
        draw_text("Data Mining View", (WIDTH // 2, 50), (255, 255, 255), center=True)

        back_btn.update(pygame.mouse.get_pos())
        back_btn.draw({"btn": (100, 100, 100), "text": (255, 255, 255), "border": (150, 150, 150)})

        kmeans_btn.update(pygame.mouse.get_pos())
        kmeans_btn.draw({"btn": (80, 80, 120), "text": (255, 255, 255), "border": (200, 200, 255)})

        fpgrowth_btn.update(pygame.mouse.get_pos())
        fpgrowth_btn.draw({"btn": (120, 80, 80), "text": (255, 255, 255), "border": (255, 200, 200)})

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if back_btn.hovered:
                    show = False
                elif kmeans_btn.hovered:
                    labels = run_kmeans()
                    data = [f"Transaction {i + 1}: Cluster {label}" for i, label in enumerate(labels)]
                    show_visualization("K-Means Clustering", data)
                elif fpgrowth_btn.hovered:
                    patterns = run_fp_growth()
                    data = [f"{', '.join(k)}: {v}" for k, v in patterns.items()]
                    show_visualization("FP-Growth Frequent Itemsets", data)

        pygame.display.flip()
        clock.tick(30)


def main():
    global current_cart, selected_items, dark_mode

    while True:
        mouse_pos = pygame.mouse.get_pos()
        colors = DARK_MODE_COLORS if dark_mode else LIGHT_MODE_COLORS
        screen.fill(colors["bg"])

        draw_text(f"Transactions: {len(transactions)}", (BUTTON_PADDING, 20), colors["text"])

        for btn in buttons:
            btn.update(mouse_pos)
            btn.draw(colors)

        toggle_dark_btn.update(mouse_pos)
        pygame.draw.circle(screen, (240, 240, 240) if not dark_mode else (200, 200, 200),
                           toggle_dark_btn.rect.center, 25)
        pygame.draw.circle(screen, colors["border"], toggle_dark_btn.rect.center, 25, 3)

        pygame.draw.rect(screen, colors["cart"], (WIDTH - CART_WIDTH, 0, CART_WIDTH, HEIGHT))
        pygame.draw.rect(screen, colors["border"], (WIDTH - CART_WIDTH, 0, CART_WIDTH, HEIGHT), 3)
        draw_text("Cart", (WIDTH - CART_WIDTH + 20, 20), colors["text"])

        for i, item in enumerate(current_cart):
            draw_text(f"• {item}", (WIDTH - CART_WIDTH + 30, 60 + i * 30), colors["text"])

        checkout_btn.update(mouse_pos)
        checkout_btn.draw(colors)

        if len(transactions) >= 5:
            mining_btn.update(mouse_pos)
            mining_btn.draw(colors)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if toggle_dark_btn.rect.collidepoint(mouse_pos):
                    dark_mode = not dark_mode
                elif checkout_btn.hovered:
                    if current_cart:
                        animate_checkout()
                        transactions.append(list(current_cart))
                        current_cart.clear()
                        for btn in buttons:
                            btn.selected = False
                        selected_items.clear()
                elif len(transactions) >= 5 and mining_btn.hovered:
                    show_data_mining_screen()
                else:
                    for btn in buttons:
                        if btn.hovered:
                            if btn.click():
                                selected_items.add(btn.text)
                                current_cart.append(btn.text)
                            else:
                                selected_items.discard(btn.text)
                                if btn.text in current_cart:
                                    current_cart.remove(btn.text)

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()

