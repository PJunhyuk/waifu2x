---- sample input on cmd : ~waifu2x# th waifu2x.lua -model_dir models/my_model -m scale -scale 2 -i images/miku_small.png -o output.png
------ means for sample

require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
require 'sys'
require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'
local alpha_util = require 'alpha_util'

torch.setdefaulttensortype('torch.FloatTensor')


local function format_output(opt, src, no)
  no = no or 1
  local name = path.basename(src)
  local e = path.extension(name)
  local basename = name:sub(0, name:len() - e:len())
  if opt.o == "(auto)" then
    return path.join(path.dirname(src), string.format("%s_%s.png", basename, opt.m))
  else
    local basename_pos = opt.o:find("%%s")
    local no_pos = opt.o:find("%%%d*d")
    if basename_pos ~= nil and no_pos ~= nil then
	    if basename_pos < no_pos then
        return string.format(opt.o, basename, no)
      else
        return string.format(opt.o, no, basename)
      end
    elseif basename_pos ~= nil then
      return string.format(opt.o, basename)
    elseif no_pos ~= nil then
      return string.format(opt.o, no)
    else
      return opt.o
    end
  end
end


local function convert_image(opt)
  local x, meta = image_loader.load_float(opt.i)
  ------ function image_loader.load_float(file) in lib/image_loader.lua
  ------ opt.i : images/miku_small.png
  ------ meta = images/miku_samll.png

  if not x then
    ------ not used
    error(string.format("failed to load image: %s", opt.i))
  end

  local alpha = meta.alpha
  local new_x = nil
  local scale_f, image_f

  if opt.tta == 1 then
    ------ opt.tta = 0(default) -> not use TTA mode
    ------ not used
    scale_f = function(model, scale, x, block_size, batch_size)
      return reconstruct.scale_tta(model, opt.tta_level, scale, x, block_size, batch_size)
    end
    image_f = function(model, x, block_size, batch_size)
      return reconstruct.image_tta(model, opt.tta_level, x, block_size, batch_size)
    end
  else
    ------ used
    ------ local reconstruct = require 'reconstruct'
    ------ refer to lib/reconstruct.lua
    scale_f = reconstruct.scale
    ------ function scale_f is in lib/reconstruct.lua
    image_f = reconstruct.image
    ------ function image_f is in lib/reconstruct.lua
  end

  opt.o = format_output(opt, opt.i)
  ------ opt.i : images/miku_small.png
  ------ opt.o : output.png
  ------ local function format_output(opt, src, no) in this file

  if opt.m == "noise" then
    ------ opt.m : scale -> not used
    local model_path = path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level))
    local model = w2nn.load_model(model_path, opt.force_cudnn)
    if not model then
      error("Load Error: " .. model_path)
    end
    local t = sys.clock()
    new_x = image_f(model, x, opt.crop_size, opt.batch_size)
    new_x = alpha_util.composite(new_x, alpha)
    if not opt.q then
      print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
    end

  elseif opt.m == "scale" then
    ------ opt.m : scale -> used
    local model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
    ------ opt.model_dir : models/my_model
    ------ opt.scale : 2
    ------ model_path = path.join("models/my_model", "scale2.0x_model.t7")
    ------ there is scale2.0x_model.t7 file in models/my_model
    local model = w2nn.load_model(model_path, opt.force_cudnn)
    ------ function w2nn.load_model(model_path, force_cudnn) in lib/w2nn.lua
    ------ opt.force_cudnn = 0
    ------ w2nn.load_model returns model

    if not model then
      ------ not used
      error("Load Error: " .. model_path)
    end

    local t = sys.clock()
    ------ t = live time
    x = alpha_util.make_border(x, alpha, reconstruct.offset_size(model))
    ------ local alpha_util = require 'alpha_util'
    ------ function alpha_util.make_border(rgb, alpha, offset) in lib/alpha_util.lua
    new_x = scale_f(model, opt.scale, x, opt.crop_size, opt.batch_size, opt.batch_size)
    ------ local scale_f = reconstruct.scale
    ------ function reconstruct.scale(model, scale, x, block_size) in lib/reconstruct.lua
    ------ new_x = scale_f(model, 2, x, 128, 1, 1)
    ------ opt.scale : 2
    ------ opt.crop_size : 128
    ------ opt.batch_size : 1
    new_x = alpha_util.composite(new_x, alpha, model)
    ------ function alpha_util.composite(rgb, alpha, model2x) in lib/alpha_util.lua
    if not opt.q then
      ------ opt.q = 0(default) -> used
      print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
      ------ print "output.png: 0.15733909606934 sec"
      ------ opt.o : output.png
    end

  elseif opt.m == "noise_scale" then
    ------ opt.m : scale -> not used
    local model_path = path.join(opt.model_dir, ("noise%d_scale%.1fx_model.t7"):format(opt.noise_level, opt.scale))
    if path.exists(model_path) then
      local scale_model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	    local t, scale_model = pcall(w2nn.load_model, scale_model_path, opt.force_cudnn)
      local model = w2nn.load_model(model_path, opt.force_cudnn)
      if not t then
        scale_model = model
      end
      local t = sys.clock()
      x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
      new_x = scale_f(model, opt.scale, x, opt.crop_size, opt.batch_size)
      new_x = alpha_util.composite(new_x, alpha, scale_model)
      if not opt.q then
        print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
      end
    else
      local noise_model_path = path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level))
      local noise_model = w2nn.load_model(noise_model_path, opt.force_cudnn)
	    local scale_model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	    local scale_model = w2nn.load_model(scale_model_path, opt.force_cudnn)
	    local t = sys.clock()
      x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
      x = image_f(noise_model, x, opt.crop_size, opt.batch_size)
      new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
      new_x = alpha_util.composite(new_x, alpha, scale_model)
      if not opt.q then
        print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
      end
    end
  elseif opt.m == "user" then
    ------ opt.m : scale -> not used
    local model_path = opt.model_path
    local model = w2nn.load_model(model_path, opt.force_cudnn)
    if not model then
      error("Load Error: " .. model_path)
    end
    local t = sys.clock()

    x = alpha_util.make_border(x, alpha, reconstruct.offset_size(model))
    if opt.scale == 1 then
      new_x = image_f(model, x, opt.crop_size, opt.batch_size)
    else
      new_x = scale_f(model, opt.scale, x, opt.crop_size, opt.batch_size)
    end
    new_x = alpha_util.composite(new_x, alpha) -- TODO: should it use model?
    if not opt.q then
      print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
    end
  else
    ------ opt.m : scale -> not used
    error("undefined method:" .. opt.method)
  end

  image_loader.save_png(opt.o, new_x, tablex.update({depth = opt.depth, inplace = true}, meta))
  ------ opt.o : output.png
  ------ new_x = alpha_util.composite(new_x, alpha, model)
  ------ function image_loader.save_png(filename, rgb, options) in lib/image_loader.lua
end


local function convert_frames(opt)
  local model_path, scale_model, t
  local noise_scale_model = {}
  local noise_model = {}
  local user_model = nil
  local scale_f, image_f
  if opt.tta == 1 then
    scale_f = function(model, scale, x, block_size, batch_size)
      return reconstruct.scale_tta(model, opt.tta_level, scale, x, block_size, batch_size)
    end
    image_f = function(model, x, block_size, batch_size)
      return reconstruct.image_tta(model, opt.tta_level, x, block_size, batch_size)
    end
  else
    scale_f = reconstruct.scale
    image_f = reconstruct.image
  end
  if opt.m == "scale" then
    model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
    scale_model = w2nn.load_model(model_path, opt.force_cudnn)
  elseif opt.m == "noise" then
    model_path = path.join(opt.model_dir, string.format("noise%d_model.t7", opt.noise_level))
    noise_model[opt.noise_level] = w2nn.load_model(model_path, opt.force_cudnn)
  elseif opt.m == "noise_scale" then
    local model_path = path.join(opt.model_dir, ("noise%d_scale%.1fx_model.t7"):format(opt.noise_level, opt.scale))
    if path.exists(model_path) then
      noise_scale_model[opt.noise_level] = w2nn.load_model(model_path, opt.force_cudnn)
      model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
      t, scale_model = pcall(w2nn.load_model, model_path, opt.force_cudnn)
      if not t then
        scale_model = noise_scale_model[opt.noise_level]
      end
    else
      model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
      scale_model = w2nn.load_model(model_path, opt.force_cudnn)
      model_path = path.join(opt.model_dir, string.format("noise%d_model.t7", opt.noise_level))
      noise_model[opt.noise_level] = w2nn.load_model(model_path, opt.force_cudnn)
    end
  elseif opt.m == "user" then
    user_model = w2nn.load_model(opt.model_path, opt.force_cudnn)
  end
  local fp = io.open(opt.l)
  if not fp then
    error("Open Error: " .. opt.l)
  end
  local count = 0
  local lines = {}
  for line in fp:lines() do
    table.insert(lines, line)
  end
  fp:close()
  for i = 1, #lines do
    local output = format_output(opt, lines[i], i)
    if opt.resume == 0 or path.exists(output) == false then
      local x, meta = image_loader.load_float(lines[i])
      if not x then
        io.stderr:write(string.format("failed to load image: %s\n", lines[i]))
      else
        local alpha = meta.alpha
        local new_x = nil
        if opt.m == "noise" then
          new_x = image_f(noise_model[opt.noise_level], x, opt.crop_size, opt.batch_size)
	        new_x = alpha_util.composite(new_x, alpha)
	      elseif opt.m == "scale" then
	        x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	        new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
	        new_x = alpha_util.composite(new_x, alpha, scale_model)
	      elseif opt.m == "noise_scale" then
	        x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	        if noise_scale_model[opt.noise_level] then
            new_x = scale_f(noise_scale_model[opt.noise_level], opt.scale, x, opt.crop_size, opt.batch_size)
          else
            x = image_f(noise_model[opt.noise_level], x, opt.crop_size, opt.batch_size)
            new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
	        end
          new_x = alpha_util.composite(new_x, alpha, scale_model)
        elseif opt.m == "user" then
	        x = alpha_util.make_border(x, alpha, reconstruct.offset_size(user_model))
	        if opt.scale == 1 then
		        new_x = image_f(user_model, x, opt.crop_size, opt.batch_size)
          else
		        new_x = scale_f(user_model, opt.scale, x, opt.crop_size, opt.batch_size)
	        end
          new_x = alpha_util.composite(new_x, alpha)
        else
          error("undefined method:" .. opt.method)
        end
        image_loader.save_png(output, new_x, tablex.update({depth = opt.depth, inplace = true}, meta))
      end
      if not opt.q then
        xlua.progress(i, #lines)
      end
      if i % 10 == 0 then
        collectgarbage()
      end
    else
      if not opt.q then
        xlua.progress(i, #lines)
      end
    end
  end
end


local function waifu2x()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text("waifu2x")
  cmd:text("Options:")
  cmd:option("-i", "images/miku_small.png", 'path to input image')
  ------ -i : images/miku_small.png
  cmd:option("-l", "", 'path to image-list.txt')
  cmd:option("-scale", 2, 'scale factor')
  ------ -scale : 2
  cmd:option("-o", "(auto)", 'path to output file')
  ------ -o : output.png
  cmd:option("-depth", 8, 'bit-depth of the output image (8|16)')
  cmd:option("-model_dir", "./models/upconv_7/art", 'path to model directory')
  ------ -model_dir : models/my_model
  cmd:option("-name", "user", 'model name for user method')
  cmd:option("-m", "noise_scale", 'method (noise|scale|noise_scale|user)')
  ------ -m : scale
  cmd:option("-method", "", 'same as -m')
  cmd:option("-noise_level", 1, '(1|2|3)')
  cmd:option("-crop_size", 128, 'patch size per process')
  cmd:option("-batch_size", 1, 'batch_size')
  cmd:option("-resume", 0, "skip existing files (0|1)")
  cmd:option("-thread", -1, "number of CPU threads")
  cmd:option("-tta", 0, 'use TTA mode. It is slow but slightly high quality (0|1)')
  cmd:option("-tta_level", 8, 'TTA level (2|4|8). A higher value makes better quality output but slow')
  cmd:option("-force_cudnn", 0, 'use cuDNN backend (0|1)')
  cmd:option("-q", 0, 'quiet (0|1)')

  local opt = cmd:parse(arg)

  if opt.method:len() > 0 then
    ------ opt.method.len() = 0 -> not used
    opt.m = opt.method
  end
  if opt.thread > 0 then
    ------ opt.thread = -1(default) -> not used
    torch.setnumthreads(opt.thread)
  end

  if cudnn then
    cudnn.fastest = true
    if opt.l:len() > 0 then
      ------ opt.l:len() = 0(default) -> not used
      cudnn.benchmark = true -- find fastest algo
    else
      ------ used
      cudnn.benchmark = false
    end
  end

  opt.force_cudnn = opt.force_cudnn == 1
  ------ opt.forcue_cudnn = 0(default) -> opt.force_cudnn = false
  opt.q = opt.q == 1
  ------ opt.q = 0(default) -> opt.q = false
  opt.model_path = path.join(opt.model_dir, string.format("%s_model.t7", opt.name))
  ------ opt.model_path = path.join("models/mymodel", "usr_model.t7")
  if string.len(opt.l) == 0 then
    ------ string.len(opt.l) == 0, so go to local function convert_image(opt)
    convert_image(opt)
  else
    convert_frames(opt)
  end
end


waifu2x()
------ run waifu2x function
