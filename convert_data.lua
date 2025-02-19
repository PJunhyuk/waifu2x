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
  ------ used
  if max_size < 0 then
    ------ used
    return src
  end
  local tries = 4
  if src:size(2) >= max_size and src:size(3) >= max_size then
    ------ unused
    local rect
    for i = 1, tries do
      local yi = torch.random(0, src:size(2) - max_size)
      local xi = torch.random(0, src:size(3) - max_size)
      rect = iproc.crop(src, xi, yi, xi + max_size, yi + max_size)
      -- ignore simple background
      if rect:float():std() >= 0 then
        ------ unused
	      break
	    end
    end
    return rect
  else
    return src
  end
end


local function crop_if_large_pair(x, y, max_size)
  ------ unused
  if max_size < 0 then
    return x, y
  end
  local scale_y = y:size(2) / x:size(2)
  local mod = 4
  assert(x:size(3) == (y:size(3) / scale_y))
  local tries = 4
  if y:size(2) > max_size and y:size(3) > max_size then
    assert(max_size % 4 == 0)
    local rect_x, rect_y
    for i = 1, tries do
      local yi = torch.random(0, y:size(2) - max_size)
      local xi = torch.random(0, y:size(3) - max_size)
      if mod then
        yi = yi - (yi % mod)
        xi = xi - (xi % mod)
      end
      rect_y = iproc.crop(y, xi, yi, xi + max_size, yi + max_size)
      rect_x = iproc.crop(y, xi / scale_y, yi / scale_y, xi / scale_y + max_size / scale_y, yi / scale_y + max_size / scale_y)
      -- ignore simple background
      if rect_y:float():std() >= 0 then
        break
      end
    end
    return rect_x, rect_y
  else
    return x, y
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
  ------ csv[1][2] : nil

  for i = 1, #csv do
    local filename = csv[i][1]
    local csv_meta = csv[i][2]

    if csv_meta and csv_meta:len() > 0 then
      ------ unused
      ------ csv_meta = nil -> unused
      csv_meta = cjson.decode(csv_meta)
    end

    if csv_meta and csv_meta.filters then
      ------ unused
      filters = csv_meta.filters
    end

    local im, meta = image_loader.load_byte(filename)
    ------ function image_loader.load_byte(file) in lib/image_loader.lua
    local skip = false
    local alpha_color = torch.random(0, 1)

    if meta and meta.alpha then
      ------ unused
      if settings.use_transparent_png then
        ------ settings.use_transparant_png = false(default) -> unused
        im = alpha_util.fill(im, meta.alpha, alpha_color)
      else
        ------ used
        skip = true
      end
    end

    if skip then
      ------ unused
      if not skip_notice then
        print("cp#6")
        io.stderr:write("skip transparent png (settings.use_transparent_png=0)\n")
        skip_notice = true
      end
    else
      ------ used!
      if csv_meta and csv_meta.x then
        ------ unused
        -- method == user
        local yy = im
        local xx, meta2 = image_loader.load_byte(csv_meta.x)
        if meta2 and meta2.alpha then
          ------ unused
          xx = alpha_util.fill(xx, meta2.alpha, alpha_color)
        end
        xx, yy = crop_if_large_pair(xx, yy, settings.max_training_image_size)
        table.insert(x, {{y = compression.compress(yy), x = compression.compress(xx)},
        {data = {filters = filters, has_x = true}}})
      else
        im = crop_if_large(im, settings.max_training_image_size)
        im = iproc.crop_mod4(im)
        local scale = 1.0
        if settings.random_half_rate > 0.0 then
          ------ unused
          ------ settings.random_half_rate = 0(default) -> unused
          scale = 2.0
        end
        if im then
          ------ used
          if im:size(2) > (settings.crop_size * scale + MARGIN) and im:size(3) > (settings.crop_size * scale + MARGIN) then
            ------ used
            ------ settings.crop_size = 48(default)
            ------ MARGIN = 32(default)
            table.insert(x, {compression.compress(im), {data = {filters = filters}}})
          else
            io.stderr:write(string.format("\n%s: skip: image is too small (%d > size).\n", filename, settings.crop_size * scale + MARGIN))
          end
        else
          io.stderr:write(string.format("\n%s: skip: load error.\n", filename))
        end
      end
    end

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
