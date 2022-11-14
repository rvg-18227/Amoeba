import os
import constants

from remi import App, gui


class AmoebaApp(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(AmoebaApp, self).__init__(*args, static_file_path={'res': res_path})

    def main(self, *userdata):
        self.amoeba_game, self.logger = userdata
        self.amoeba_game.set_app(self)

        self.turn = 0

        mainContainer = gui.Container(
            style={'width': '100%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'
        mainContainer.set_layout_orientation(gui.Container.LAYOUT_HORIZONTAL)

        boardContainer = gui.Container(
            style={'width': '55%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        boardContainer.style['justify-content'] = 'center'
        boardContainer.style['align-items'] = 'center'

        menuContainer = gui.Container(
            style={'width': '45%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        menuContainer.style['justify-content'] = 'center'
        menuContainer.style['align-items'] = 'center'

        header_label = gui.Label("Project 4: Amoeba", style={'font-size': '36px', 'font-weight': 'bold'})
        menuContainer.append(header_label)

        bt_hbox = gui.HBox()
        go_start_bt = gui.Button("Back to Start")
        prev_turn_bt = gui.Button("Previous Turn")
        next_turn_bt = gui.Button("Next Turn")
        go_end_bt = gui.Button("Skip to End")

        bt_hbox.append([go_start_bt, prev_turn_bt, next_turn_bt, go_end_bt])

        go_start_bt.onclick.do(self.go_start_bt_press)
        prev_turn_bt.onclick.do(self.prev_turn_bt_press)
        next_turn_bt.onclick.do(self.next_turn_bt_press)
        go_end_bt.onclick.do(self.go_end_bt_press)

        menuContainer.append(bt_hbox)

        turn_label = gui.Label("Press a button or choose a turn: ", style={'margin': '5px'})

        ch_hbox = gui.HBox()
        self.view_drop_down = gui.DropDown(style={'padding': '5px', 'text-align': 'center'})
        for i in range(self.amoeba_game.game_end + 1):
            self.view_drop_down.append("Turn {}".format(i), i)

        self.view_drop_down.onchange.do(self.view_drop_down_changed)

        self.label = gui.Label("Starting state.", style={'margin': '5px auto'})

        ch_hbox.append([turn_label, self.view_drop_down, self.label])

        menuContainer.append(gui.Label())
        menuContainer.append(ch_hbox)

        self.frame = gui.Image(r'/res:render\0.png', width=constants.vis_width, height=constants.vis_height,
                               margin='1px')

        boardContainer.append(self.frame)
        mainContainer.append(boardContainer)
        mainContainer.append(menuContainer)

        return mainContainer

    def go_start_bt_press(self, widget):
        self.label.set_text("Processing...")
        self.do_gui_update()
        self.display_frame(0)

    def prev_turn_bt_press(self, widget):
        if self.turn != 0:
            self.label.set_text("Processing...")
            self.do_gui_update()
            self.display_frame(self.turn - 1)

    def next_turn_bt_press(self, widget):
        if self.turn != self.amoeba_game.game_end:
            self.label.set_text("Processing...")
            self.do_gui_update()
            self.display_frame(self.turn + 1)

    def go_end_bt_press(self, widget):
        self.label.set_text("Processing...")
        self.do_gui_update()
        self.display_frame(self.amoeba_game.game_end)

    def view_drop_down_changed(self, widget, value):
        turn = widget.get_key()
        if 0 <= turn <= self.amoeba_game.game_end:
            self.label.set_text("Processing...")
            self.do_gui_update()
            self.display_frame(turn)

    def display_frame(self, turn):
        self.turn = turn
        self.frame.set_image(r'/res:render\\' + str(self.turn) + r'.png')
        self.view_drop_down.select_by_key(turn)
        if self.turn == self.amoeba_game.max_turns:
            self.label.set_text("Goal size not achieved.")
        elif self.turn == self.amoeba_game.game_end:
            self.label.set_text("Goal size achieved!")
        elif self.turn == 0:
            self.label.set_text("Starting state.")
        else:
            self.label.set_text("In progress...")

        self.do_gui_update()
