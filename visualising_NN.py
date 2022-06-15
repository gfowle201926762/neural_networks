import pygame
pygame.init()

import final_nnfs as nn


class Visualisation:
    def __init__(self, layers, all_layers, max_height, no_depth):
        self.grey = (70, 70, 70)
        self.black = (0, 0, 0)
        self.white = (255, 253, 209)
        self.red = (199, 21, 133)
        self.blue = (30, 144, 255)

        self.layers = layers
        self.all_layers = all_layers

        self.dis_height = 800
        self.dis_width = 1400

        self.height = 700
        self.width = 1000

        self.height_padding = (self.dis_height - self.height) / 2
        self.width_padding = (self.dis_width - self.width) / 2
        self.circle_sizer = (1/25) * (max_height * no_depth) ** 1/3
        self.circle_width = min(25, (25 / self.circle_sizer))

        self.font = pygame.font.SysFont("arial", 20)
        self.small_font = pygame.font.SysFont("arial", 14)
        self.clock = pygame.time.Clock()

        self.dis = pygame.display.set_mode((self.dis_width, self.dis_height))
        pygame.display.set_caption("neural networks")

    def draw_nodes(self, label_names, input_names):
        for w in range(1, len(self.layers) + 1):
            for h in range(1, self.layers[w-1] + 1):
                right = w * (self.width / len(self.layers)) - 0.5 * (self.width / len(self.layers)) + self.width_padding
                down = h * (self.height / self.layers[w-1]) - 0.5 * (self.height / self.layers[w-1]) + self.height_padding
                if w > 1:
                    bias = self.all_layers[w - 2].biases[h-1]
                    a = 200 * (abs(bias) ** 1/2)
                    if a >= 125:
                        a = 125

                    if bias >= 0:
                        line_colour = (125 + a, 100 - (a / 2), 125 - a)
                    if bias < 0:
                        line_colour = (125 - a, 100 - (a / 2), 125 + a)

                    

                    pygame.draw.circle(self.dis, line_colour, (right, down), self.circle_width)
                    pygame.draw.circle(self.dis, self.black, (right, down), self.circle_width - self.circle_width / 5)
                    print(self.circle_width)

                    if w == len(self.layers):
                        text1 = self.small_font.render(f"{label_names[h-1]}", True, self.white)
                        text_rect1 = text1.get_rect(midleft=(right + self.circle_width + 5, down))
                        self.dis.blit(text1, text_rect1)                        

                else:
                    pygame.draw.circle(self.dis, self.black, (right, down), self.circle_width)
                    text1 = self.small_font.render(f"{input_names[h-1]}", True, self.white)
                    text_rect1 = text1.get_rect(midright=(right - self.circle_width - 5, down))
                    self.dis.blit(text1, text_rect1)

    def draw_connections(self):
        # go backwards to avoid inputs.
        for w in range(2, len(self.layers) + 1):
            for h in range(1, self.layers[w-1] + 1):
                # a single node in the current layer, 
                right = w * (self.width / len(self.layers)) - 0.5 * (self.width / len(self.layers)) + self.width_padding
                down = h * (self.height / self.layers[w-1]) - 0.5 * (self.height / self.layers[w-1]) + self.height_padding
                start = (right, down)

                prev_w = w - 1
                for prev_h in range(1, self.layers[w - 2] + 1):
                    end = (prev_w * (self.width / len(self.layers)) - 0.5 * (self.width / len(self.layers)) + self.width_padding, prev_h * (self.height / self.layers[w - 2]) - 0.5 * (self.height / self.layers[w - 2]) + self.height_padding)
                                        
                    weight = self.all_layers[w-2].weights[:, h - 1][prev_h - 1]

                    a = 200 * (abs(weight) ** 1/2)
                    if a >= 125:
                        a = 125

                    if weight >= 0:
                        line_colour = (125 + a, 100 - (a / 2), 125 - a)
                    if weight < 0:
                        line_colour = (125 - a, 100 - (a / 2), 125 + a)

                    pygame.draw.line(self.dis, line_colour, start, end)
               

    def initialise(self, label_names, input_names):
        self.dis.fill(self.grey)
        self.draw_connections()
        self.draw_nodes(label_names, input_names)

    def update(self, loss, accuracy, epoch, hidden_activation, output_activation, label_names, input_names):
        self.initialise(label_names, input_names)
        text1 = self.font.render(f"EPOCH: {epoch}", True, self.white)
        text2 = self.font.render(f"LOSS: {loss:.4f}", True, self.white)
        text3 = self.font.render(f"ACC: {accuracy:.4f}", True, self.white)
        text4 = self.font.render(f"{hidden_activation}", True, self.white)
        text5 = self.font.render(f"{output_activation}", True, self.white)
        text_rect1 = text1.get_rect(topright=(self.dis_width - 10, 10))
        text_rect2 = text2.get_rect(topright=(self.dis_width - 10, 60))
        text_rect3 = text3.get_rect(topright=(self.dis_width - 10, 110))
        text_rect4 = text4.get_rect(topright=(self.dis_width - 10, self.dis_height - 50))
        text_rect5 = text5.get_rect(topright=(self.dis_width - 10, self.dis_height - 100))
        self.dis.blit(text1, text_rect1)
        self.dis.blit(text2, text_rect2)
        self.dis.blit(text3, text_rect3)
        self.dis.blit(text4, text_rect4)
        self.dis.blit(text5, text_rect5)



    def play(self, epochs):
        game_over = False
        x = 0
        pause = False
        begin = True

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if pause == True:
                            pause = False
                        if pause == False:
                            pause = True
            if begin == True:
                self.initialise(nn.label_names, nn.input_names)
                begin = False
            if x <= epochs and pause == False:
                loss, accuracy, epoch = nn.model.vis_go(x, nn.inputs, nn.y)
                if x % 100 == 0:
                    self.update(loss, accuracy, epoch, nn.model.hidden_activation, nn.model.output_activation, nn.label_names, nn.input_names)
                x += 1
                self.clock.tick(200)

            pygame.display.update()



nn.model.generate_vis()
vis = Visualisation(nn.model.vis_layers, nn.model.layers, nn.model.max_height, nn.model.no_depth)
vis.play(10000)

