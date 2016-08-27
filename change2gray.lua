local Image = require 'image'

local img = Image.load('./output.png', 3, 'byte')
img = Image.rgb2y(img)
img = Image.save('./output-gray.png')
