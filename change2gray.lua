local Image = require 'image'

local img = Image.load('./miku_small.png', 3, 'byte')
img = Image.rgb2y(img)
Image.savePNG('./input-gray.png', img)
