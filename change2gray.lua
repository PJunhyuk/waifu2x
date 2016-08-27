local Image = require 'image'

local img = Image.load('./output.png', 3, 'byte')
img = Image.rgb2y(img)
Image.savePNG('./output-gray.png', img)
