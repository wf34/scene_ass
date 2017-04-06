
import colorsys
import numpy as np

class color_generator:
  def __init__(self, colors_amount):
    assert isinstance(colors_amount, int)
    assert colors_amount > 0 and colors_amount < 36
    self.colors_amount = colors_amount
    self.color_index = 0
    np.random.seed(34)


  def get_next_color(self):
    indices = np.arange(0., 360., 360. / self.colors_amount)
    i = indices[self.color_index]
    self.color_index += 1
    hue = i/360.
    lightness = (50 + np.random.rand() * 10)/100.
    saturation = (90 + np.random.rand() * 10)/100.
    return colorsys.hls_to_rgb(hue, lightness, saturation)
