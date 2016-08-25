---- sample input on cmd : ~/waifu2x# th convert_data.lua
------ means for sample

require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

require 'image'
local cjson = require 'cjson'
local csvigo = require 'csvigo'
local compression = require 'compression'
local settings = require 'settings'
local image_loader = require 'image_loader'
local iproc = require 'iproc'
local alpha_util = require 'alpha_util'

local function crop_if_large(src, max_size)
  if max_size < 0 then
    return src
  end
  local tries = 4
  if src:size(2) >= max_size and src:size(3) >= max_size then
    local rect
    for i = 1, tries do
      local yi = torch.random(0, src:size(2) - max_size)
      local xi = torch.random(0, src:size(3) - max_size)
      rect = iproc.crop(src, xi, yi, xi + max_size, yi + max_size)
      -- ignore simple background
      if rect:float():std() >= 0 then
	      break
	    end
    end
    return rect
  else
    return src
  end
end

local function load_images(list)
  local MARGIN = 32
  local csv = csvigo.load({path = list, verbose = false, mode = "raw"})
  ------ csv : comma-separated values
  local x = {}
  local skip_notice = false
  ------ #csv : 9999
  ------ csv[1][1] : /CelebA/Img/img_align_celeba/Img/000755.jpg

  for i = 1, #csv do
    local filename = csv[i][1]
    local im, meta = image_loader.load_byte(filename)
    ------ function image_loader.load_byte(file) in lib/image_loader.lua
    local alpha_color = torch.random(0, 1)
    im = crop_if_large(im, settings.max_training_image_size)
    im = iproc.crop_mod4(im)
    local scale = 1.0
    table.insert(x, {compression.compress(im), {data = {filters = filters}}})
    xlua.progress(i, #csv)
    if i % 10 == 0 then
      ------ used in case
      collectgarbage()
    end
  end
  return x
end

torch.manualSeed(settings.seed)
print(settings)
------ sample result
--[[
{
  active_cropping_rate : 0.5
  batch_size : 16
  name : "user"
  method : "scale"
  max_size : 256
  validation_crops : 200
  plot : false
  patches : 64
  save_history : false
  resize_blur_max : 1.05
  gpu : -1
  test : "images/miku_small.png"
  downsampling_filters :
  {
    1 : "Box"
    2 : "Lanczos"
    3 : "Sinc"
  }
  resume : ""
  crop_size : 48
  random_color_noise_rate : 0
  images : ./data/images.t7"
  seed : 11
  image_list : "./data/image_list.txt"
  resize_blur_min : 0.95
  nr_rate : 0.65
  model_file : "./models/scale2.0x_model.t7"
  learning_rate_decay : 3e-07
  model : "vgg_7"
  active_cropping_tries : 10
  use_transparent_png : false
  oracle_drop_rate : 0.5
  inner_epoch : 4
  random_overlay_rate : 0
  epoch : 50
  data_dir : "./data"
  learning_rate : 0.00025
  random_half_rate : 0
  scale : 2
  jpeg_chroma_subsampling_rate : 0.5
  max_training_image_size : -1
  oracle_rate : 0.1
  model_dir : "./models"
  style : "art"
  random_unsharp_mask_rate : 0
  color : "rgb"
  noise_level : 1
  backend : "cunn"
  validation_rate : 0.05
  thread : -1
}
 [=================== 202599/202599 ===================]  Tot:  1h36m | Step: 42ms
]]
local x = load_images(settings.image_list)
torch.save(settings.images, x)
