---- sample input on cmd : ~/waifu2x# th train.lua -model upconv_7 -model_dir models/my_model -method scale -scale 2 -test images/miku_small.png -epoch 2 -inner_epoch 2
------ means for sample

require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
require 'optim'
require 'xlua'
require 'w2nn'
local settings = require 'settings'
local srcnn = require 'srcnn'
local minibatch_adam = require 'minibatch_adam'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local compression = require 'compression'
local pairwise_transform = require 'pairwise_transform'
local image_loader = require 'image_loader'

local function save_test_scale(model, rgb, file)
  local up = reconstruct.scale(model, settings.scale, rgb)
  image.save(file, up)
end

local function save_test_jpeg(model, rgb, file)
  local im, count = reconstruct.image(model, rgb)
  image.save(file, im)
end

local function save_test_user(model, rgb, file)
  if settings.scale == 1 then
    save_test_jpeg(model, rgb, file)
  else
    save_test_scale(model, rgb, file)
  end
end

local function split_data(x, test_size)
  local index = torch.randperm(#x)
  local train_size = #x - test_size
  local train_x = {}
  local valid_x = {}
  for i = 1, train_size do
    train_x[i] = x[index[i]]
  end
  for i = 1, test_size do
    valid_x[i] = x[index[train_size + i]]
  end
  return train_x, valid_x
end

local function make_validation_set(x, transformer, n, patches)
  n = n or 4
  local validation_patches = math.min(16, patches or 16)
  local data = {}
  for i = 1, #x do
    for k = 1, math.max(n / validation_patches, 1) do
      local xy = transformer(x[i], true, validation_patches)
      for j = 1, #xy do
        table.insert(data, {x = xy[j][1], y = xy[j][2]})
      end
    end
    xlua.progress(i, #x)
    collectgarbage()
  end
  local new_data = {}
  local perm = torch.randperm(#data)
  for i = 1, perm:size(1) do
    new_data[i] = data[perm[i]]
  end
  data = new_data
  return data
end

local function validate(model, criterion, eval_metric, data, batch_size)
  local loss = 0
  local mse = 0
  local loss_count = 0
  local inputs_tmp = torch.Tensor(batch_size, data[1].x:size(1), data[1].x:size(2), data[1].x:size(3)):zero()
  local targets_tmp = torch.Tensor(batch_size, data[1].y:size(1), data[1].y:size(2), data[1].y:size(3)):zero()
  local inputs = inputs_tmp:clone():cuda()
  local targets = targets_tmp:clone():cuda()
  for t = 1, #data, batch_size do
    if t + batch_size -1 > #data then
      break
    end
    for i = 1, batch_size do
      inputs_tmp[i]:copy(data[t + i - 1].x)
      targets_tmp[i]:copy(data[t + i - 1].y)
    end
    inputs:copy(inputs_tmp)
    targets:copy(targets_tmp)
    local z = model:forward(inputs)
    loss = loss + criterion:forward(z, targets)
    mse = mse + eval_metric:forward(z, targets)
    loss_count = loss_count + 1
    if loss_count % 10 == 0 then
      xlua.progress(t, #data)
      collectgarbage()
    end
  end
  xlua.progress(#data, #data)
  return {loss = loss / loss_count, MSE = mse / loss_count, PSNR = 10 * math.log10(1 / (mse / loss_count))}
end

local function create_criterion(model)
  if reconstruct.is_rgb(model) then
    local offset = reconstruct.offset_size(model)
    local output_w = settings.crop_size - offset * 2
    local weight = torch.Tensor(3, output_w * output_w)
    weight[1]:fill(0.29891 * 3) -- R
    weight[2]:fill(0.58661 * 3) -- G
    weight[3]:fill(0.11448 * 3) -- B
    return w2nn.ClippedWeightedHuberCriterion(weight, 0.1, {0.0, 1.0}):cuda()
  else
    local offset = reconstruct.offset_size(model)
    local output_w = settings.crop_size - offset * 2
    local weight = torch.Tensor(1, output_w * output_w)
    weight[1]:fill(1.0)
    return w2nn.ClippedWeightedHuberCriterion(weight, 0.1, {0.0, 1.0}):cuda()
  end
end

local function transformer(model, x, is_validation, n, offset)
  local meta = {data = {}}
  local y = nil
  if type(x) == "table" and type(x[2]) == "table" then
    meta = x[2]
    if x[1].x and x[1].y then
      y = compression.decompress(x[1].y)
      x = compression.decompress(x[1].x)
    else
      x = compression.decompress(x[1])
    end
  else
    x = compression.decompress(x)
  end
  n = n or settings.patches
  if is_validation == nil then is_validation = false end
  local random_color_noise_rate = nil
  local random_overlay_rate = nil
  local active_cropping_rate = nil
  local active_cropping_tries = nil
  if is_validation then
    active_cropping_rate = settings.active_cropping_rate
    active_cropping_tries = settings.active_cropping_tries
    random_color_noise_rate = 0.0
    random_overlay_rate = 0.0
  else
    active_cropping_rate = settings.active_cropping_rate
    active_cropping_tries = settings.active_cropping_tries
    random_color_noise_rate = settings.random_color_noise_rate
    random_overlay_rate = settings.random_overlay_rate
  end
  if settings.method == "scale" then
    local conf = tablex.update({
        downsampling_filters = settings.downsampling_filters,
        random_half_rate = settings.random_half_rate,
        random_color_noise_rate = random_color_noise_rate,
        random_overlay_rate = random_overlay_rate,
        random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
        max_size = settings.max_size,
        active_cropping_rate = active_cropping_rate,
        active_cropping_tries = active_cropping_tries,
        rgb = (settings.color == "rgb"),
        x_upsampling = not reconstruct.has_resize(model),
        resize_blur_min = settings.resize_blur_min,
        resize_blur_max = settings.resize_blur_max}, meta)
    return pairwise_transform.scale(x, settings.scale, settings.crop_size, offset, n, conf)
  elseif settings.method == "noise" then
    local conf = tablex.update({
        random_half_rate = settings.random_half_rate,
        random_color_noise_rate = random_color_noise_rate,
        random_overlay_rate = random_overlay_rate,
        random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
        max_size = settings.max_size,
        jpeg_chroma_subsampling_rate = settings.jpeg_chroma_subsampling_rate,
        active_cropping_rate = active_cropping_rate,
        active_cropping_tries = active_cropping_tries,
        nr_rate = settings.nr_rate,
        rgb = (settings.color == "rgb")}, meta)
    return pairwise_transform.jpeg(x,
      settings.style,
      settings.noise_level,
      settings.crop_size, offset,
      n, conf)
  elseif settings.method == "noise_scale" then
    local conf = tablex.update({
        downsampling_filters = settings.downsampling_filters,
        random_half_rate = settings.random_half_rate,
        random_color_noise_rate = random_color_noise_rate,
        random_overlay_rate = random_overlay_rate,
        random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
        max_size = settings.max_size,
        jpeg_chroma_subsampling_rate = settings.jpeg_chroma_subsampling_rate,
        nr_rate = settings.nr_rate,
        active_cropping_rate = active_cropping_rate,
        active_cropping_tries = active_cropping_tries,
        rgb = (settings.color == "rgb"),
        x_upsampling = not reconstruct.has_resize(model),
        resize_blur_min = settings.resize_blur_min,
        resize_blur_max = settings.resize_blur_max}, meta)
    return pairwise_transform.jpeg_scale(x,
      settings.scale,
      settings.style,
      settings.noise_level,
      settings.crop_size, offset,
      n, conf)
  elseif settings.method == "user" then
    local conf = tablex.update({
        max_size = settings.max_size,
        active_cropping_rate = active_cropping_rate,
        active_cropping_tries = active_cropping_tries,
        rgb = (settings.color == "rgb")}, meta)
    return pairwise_transform.user(x, y,
      settings.crop_size, offset,
      n, conf)
  end
end

local function resampling(x, y, train_x, transformer, input_size, target_size)
  local c = 1
  local shuffle = torch.randperm(#train_x)
  for t = 1, #train_x do
    xlua.progress(t, #train_x)
    local xy = transformer(train_x[shuffle[t]], false, settings.patches)
    for i = 1, #xy do
      x[c]:copy(xy[i][1])
      y[c]:copy(xy[i][2])
      c = c + 1
      if c > x:size(1) then
        break
      end
    end
    if c > x:size(1) then
      break
    end
    if t % 50 == 0 then
      collectgarbage()
    end
  end
  xlua.progress(#train_x, #train_x)
end

local function get_oracle_data(x, y, instance_loss, k, samples)
  local index = torch.LongTensor(instance_loss:size(1))
  local dummy = torch.Tensor(instance_loss:size(1))
  torch.topk(dummy, index, instance_loss, k, 1, true)
  print("MSE of all data: " ..instance_loss:mean() .. ", MSE of oracle data: " .. dummy:mean())
  local shuffle = torch.randperm(k)
  local x_s = x:size()
  local y_s = y:size()
  x_s[1] = samples
  y_s[1] = samples
  local oracle_x = torch.Tensor(table.unpack(torch.totable(x_s)))
  local oracle_y = torch.Tensor(table.unpack(torch.totable(y_s)))

  for i = 1, samples do
    oracle_x[i]:copy(x[index[shuffle[i]]])
    oracle_y[i]:copy(y[index[shuffle[i]]])
  end
  return oracle_x, oracle_y
end

local function remove_small_image(x)
  local new_x = {}
  for i = 1, #x do
    local xe, meta, x_s
    xe = x[i]
    if type(x) == "table" and type(x[2]) == "table" then
      if xe[1].x and xe[1].y then
        x_s = compression.size(xe[1].y) -- y size
      else
        x_s = compression.size(xe[1])
      end
    else
      x_s = compression.size(xe)
    end
    if x_s[2] / settings.scale > settings.crop_size + 32 and
    x_s[3] / settings.scale > settings.crop_size + 32 then
      table.insert(new_x, x[i])
    end
    if i % 100 == 0 then
      collectgarbage()
    end
  end
  print(string.format("%d small images are removed", #x - #new_x))
  ------ print "0 small images are removed"
  return new_x
end

local function plot(train, valid)
  gnuplot.plot({
      {'training', torch.Tensor(train), '-'},
      {'validation', torch.Tensor(valid), '-'}})
end

local function train()
  local hist_train = {}
  local hist_valid = {}
  local model
  ------ used
  if settings.resume:len() > 0 then
    ------ unused
    ------ on lib/settings.lua
    ------ resume = ""(default) -> unused
    model = torch.load(settings.resume, "ascii")
  else
    ------ used
    model = srcnn.create(settings.model, settings.backend, settings.color)
    ------ function srcnn.create(model_name, backend, color) in lib/srcnn.lua
    ------ settings.model = "upconv_7"
    ------ settings.backend = "cunn"
    ------ settings.color = "rgb"
  end
  local offset = reconstruct.offset_size(model)
  local pairwise_func = function(x, is_validation, n)
    ------ used -> very often
    return transformer(model, x, is_validation, n, offset)
    ------ local function transformer(model, x, is_validation, n, offset) in this file
  end
  local criterion = create_criterion(model)
  local eval_metric = w2nn.ClippedMSECriterion(0, 1):cuda()
  local x = remove_small_image(torch.load(settings.images))
  ------ local function remove_small_image(x) in this file
  ------ print "0 small images are removed"
  local train_x, valid_x = split_data(x, math.max(math.floor(settings.validation_rate * #x), 1))
  local adam_config = {
    xLearningRate = settings.learning_rate,
    xBatchSize = settings.batch_size,
    xLearningRateDecay = settings.learning_rate_decay
  }
  local ch = nil
  if settings.color == "y" then
    ch = 1
  elseif settings.color == "rgb" then
    ------ used
    ch = 3
  end
  local best_score = 1000.0
  print("# make validation-set")
  ------ print "# make validation-set"
  local valid_xy = make_validation_set(valid_x, pairwise_func,
    settings.validation_crops,
    settings.patches)
  valid_x = nil

  collectgarbage()
  model:cuda()
  print("load .. " .. #train_x)
  ------ #train_x = 9500
  ------ print "load .. 9500"

  local x = nil
  local y = torch.Tensor(settings.patches * #train_x,
    ch * (settings.crop_size - offset * 2) * (settings.crop_size - offset * 2)):zero()

  if reconstruct.has_resize(model) then
    ------ used
    x = torch.Tensor(settings.patches * #train_x,
      ch, settings.crop_size / settings.scale, settings.crop_size / settings.scale)
  else
    x = torch.Tensor(settings.patches * #train_x,
      ch, settings.crop_size, settings.crop_size)
  end

  local instance_loss = nil

  for epoch = 1, settings.epoch do
    ------ used
    model:training()
    print("# " .. epoch)
    if adam_config.learningRate then
      print("cp#7")
      print("learning rate: " .. adam_config.learningRate)
      ------ non-existent on # 1 / existent on # 2
      ------ on # 2 -> print "learning rate: 0.00018317702227433"
    end
    print("## resampling")
    ------ print "## resampling"
    if instance_loss then
      print("cp#8")
      ------ instance_loss = nil -> (maybe) not used
      -- active learning
      local oracle_k = math.min(x:size(1) * (settings.oracle_rate * (1 / (1 - settings.oracle_drop_rate))), x:size(1))
      local oracle_n = math.min(x:size(1) * settings.oracle_rate, x:size(1))
      if oracle_n > 0 then
        print("cp#9")
        local oracle_x, oracle_y = get_oracle_data(x, y, instance_loss, oracle_k, oracle_n)
        resampling(x:narrow(1, oracle_x:size(1) + 1, x:size(1)-oracle_x:size(1)), y:narrow(1, oracle_x:size(1) + 1, x:size(1) - oracle_x:size(1)), train_x, pairwise_func)
        x:narrow(1, 1, oracle_x:size(1)):copy(oracle_x)
        y:narrow(1, 1, oracle_y:size(1)):copy(oracle_y)

        local draw_n = math.floor(math.sqrt(oracle_x:size(1), 0.5))
        if draw_n > 100 then
          print("cp#10")
          draw_n = 100
        end
        image.save(path.join(settings.model_dir, "oracle_x.png"),
          image.toDisplayTensor({ input = oracle_x:narrow(1, 1, draw_n * draw_n), padding = 2, nrow = draw_n, min = 0, max = 1 }))
      else
        print("cp#11")
        resampling(x, y, train_x, pairwise_func)
      end
    else
      resampling(x, y, train_x, pairwise_func)
    end

    collectgarbage()
    instance_loss = torch.Tensor(x:size(1)):zero()

    for i = 1, settings.inner_epoch do
      ------ used
      ------ settings.inner_epoch = 2(set)
      model:training()
      local train_score, il = minibatch_adam(model, criterion, eval_metric, x, y, adam_config)
      instance_loss:copy(il)
      print(train_score)
      model:evaluate()
      print("# validation")
      local score = validate(model, criterion, eval_metric, valid_xy, adam_config.xBatchSize)
      table.insert(hist_train, train_score.loss)
      table.insert(hist_valid, score.loss)
      if settings.plot then
        print("cp#13")
        plot(hist_train, hist_valid)
      end
      if score.MSE < best_score then
        ------ used
        local test_image = image_loader.load_float(settings.test) -- reload
        best_score = score.MSE
        print("* Best model is updated")
        if settings.save_history then
          ------ unused
          ------ settings.save_history = "fault"(default) -> unused
          torch.save(settings.model_file_best, model:clearState(), "ascii")
          torch.save(string.format(settings.model_file, epoch, i), model:clearState(), "ascii")
          if settings.method == "noise" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("noise%d_best.%d-%d.png"):format(settings.noise_level,
                epoch, i))
            save_test_jpeg(model, test_image, log)
          elseif settings.method == "scale" then
            ------ settings.method = "scale"(set) -> used
            local log = path.join(settings.model_dir,
              ("scale%.1f_best.%d-%d.png"):format(settings.scale,
                epoch, i))
            save_test_scale(model, test_image, log)
          elseif settings.method == "noise_scale" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("noise%d_scale%.1f_best.%d-%d.png"):format(settings.noise_level,
                settings.scale,
                epoch, i))
            save_test_scale(model, test_image, log)
          elseif settings.method == "user" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("%s_best.%d-%d.png"):format(settings.name,
                epoch, i))
            save_test_user(model, test_image, log)
          end
        else
          torch.save(settings.model_file, model:clearState(), "ascii")
          if settings.method == "noise" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("noise%d_best.png"):format(settings.noise_level))
            save_test_jpeg(model, test_image, log)

          elseif settings.method == "scale" then
            ------ settings.method = "scale"(set) -> used
            local log = path.join(settings.model_dir,
              ("scale%.1f_best.png"):format(settings.scale))
            save_test_scale(model, test_image, log)

          elseif settings.method == "noise_scale" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("noise%d_scale%.1f_best.png"):format(settings.noise_level,
                settings.scale))
            save_test_scale(model, test_image, log)
          elseif settings.method == "user" then
            ------ settings.method = "scale"(set) -> not used
            local log = path.join(settings.model_dir,
              ("%s_best.png"):format(settings.name))
            save_test_user(model, test_image, log)
          end
        end
      end
      print("Batch-wise PSNR: " .. score.PSNR .. ", loss: " .. score.loss .. ", MSE: " .. score.MSE .. ", Minimum MSE: " .. best_score)
      ------ Batch-wise PSNR: 31.077134350816, loss: 0.00044024189632818, MSE: 0.00078034484351066, Minimum MSE: 0.00078034484351066
      collectgarbage()
    end
  end
end

if settings.gpu > 0 then
  ------ unused
  print("cp#16")
  cutorch.setDevice(settings.gpu)
end

torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
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
  images : "./data/images.t7"
  seed : 11
  image_list : "./data/image_list.txt"
  resize_blur_min : 0.95
  nr_rate : 0.65
  model_file : "models/my_model/scale2.0x_model.t7"
  learning_rate_decay : 3e-07
  model : "upconv_7"
  active_cropping_tries : 10
  use_transparent_png : false
  oracle_drop_rate : 0.5
  inner_epoch : 2
  random_overlay_rate : 0
  epoch : 2
  data_dir : "./data"
  learning_rate : 0.00025
  random_half_rate : 0
  scale : 2
  jpeg_chroma_subsampling_rate : 0.5
  max_training_image_size : -1
  oracle_rate : 0.1
  model_dir : "models/my_model"
  style : "art"
  random_unsharp_mask_rate : 0
  color : "rgb"
  noise_level : 1
  backend : "cunn"
  validation_rate : 0.05
  thread : -1
}
]]
train()
------ local function train() in this file
