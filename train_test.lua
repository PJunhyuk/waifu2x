------ means for sample

require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

------ for train
require 'optim'
require 'xlua'
require 'w2nn'
------ for convert_data
require 'image'

------ for train
local settings = require 'settings'
local srcnn = require 'srcnn'
local minibatch_adam = require 'minibatch_adam'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local compression = require 'compression'
local pairwise_transform = require 'pairwise_transform'
local image_loader = require 'image_loader'
------ for convert_data
local cjson = require 'cjson'
local csvigo = require 'csvigo'
local alpha_util = require 'alpha_util'

local function save_test_scale(model, rgb, file)
  ------ used
  local up = reconstruct.scale(model, settings.scale, rgb)
  image.save(file, up)
end

local function split_data(x, test_size)
  ------ used
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
  ------ used
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
  ------ used
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
  ------ used
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
  ------ used very often
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
end

local function resampling(x, y, train_x, transformer, input_size, target_size)
  ------ used
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

local function remove_small_image(x)
  ------ used
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
    if x_s[2] / settings.scale > settings.crop_size + 32 and x_s[3] / settings.scale > settings.crop_size + 32 then
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





---- code start : convert_data + train
------ initial setting
torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
print(settings)

------ variation setting for training
local hist_train = {}
local hist_valid = {}
local model = srcnn.create(settings.model, settings.backend, settings.color)
local offset = reconstruct.offset_size(model)
local criterion = create_criterion(model)
local eval_metric = w2nn.ClippedMSECriterion(0, 1):cuda()
local pairwise_func = function(x, is_validation, n)
  return transformer(model, x, is_validation, n, offset)
end

------ for convert_data
local csv = csvigo.load({path = settings.image_list, verbose = false, mode = "raw"}) ------ csv : comma-separated values
local x = {}
------ #csv : 9999 ||| csv[1][1] : /CelebA/Img/img_align_celeba/Img/000755.jpg
for i = 1, #csv do
  local filename = csv[i][1]
  local im, meta = image_loader.load_byte(filename)
  im = image.rgb2y(im) ------ rgb to gray
  im = iproc.crop_mod4(im) ------ cut image mod 4
  ------ table inserting
  table.insert(x, {compression.compress(im), {data = {filters = filters}}})
  xlua.progress(i, #csv)
  if i % 10 == 0 then
    collectgarbage()
  end
end

x = remove_small_image(x)
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
local valid_xy = make_validation_set(valid_x, pairwise_func, settings.validation_crops, settings.patches)
valid_x = nil
collectgarbage()
model:cuda()
print("load .. " .. #train_x)
------ #train_x = 9500
------ print "load .. 9500"

local x = nil
local y = torch.Tensor(settings.patches * #train_x, ch * (settings.crop_size - offset * 2) * (settings.crop_size - offset * 2)):zero()
x = torch.Tensor(settings.patches * #train_x, ch, settings.crop_size / settings.scale, settings.crop_size / settings.scale)
local instance_loss = nil

------ delete settings.epoch for no FOR
model:training()

print("## resampling")
------ print "## resampling"

resampling(x, y, train_x, pairwise_func)
collectgarbage()
instance_loss = torch.Tensor(x:size(1)):zero()

------ delete settings.inner_epoch for no FOR
model:training()
local train_score, il = minibatch_adam(model, criterion, eval_metric, x, y, adam_config)
instance_loss:copy(il)
print(train_score)
model:evaluate()
print("# validation")
local score = validate(model, criterion, eval_metric, valid_xy, adam_config.xBatchSize)
table.insert(hist_train, train_score.loss)
table.insert(hist_valid, score.loss)
if score.MSE < best_score then
  ------ used
  local test_image = image_loader.load_float(settings.test) -- reload
  best_score = score.MSE
  print("* Best model is updated")
  ------ settings.save_history = "fault"(default) -> unused
  torch.save(settings.model_file, model:clearState(), "ascii")
  ------ settings.method = "scale"(set) -> used
  local log = path.join(settings.model_dir, ("scale%.1f_best.png"):format(settings.scale))
  save_test_scale(model, test_image, log)
end
print("Batch-wise PSNR: " .. score.PSNR .. ", loss: " .. score.loss .. ", MSE: " .. score.MSE .. ", Minimum MSE: " .. best_score)
------ Batch-wise PSNR: 31.077134350816, loss: 0.00044024189632818, MSE: 0.00078034484351066, Minimum MSE: 0.00078034484351066
collectgarbage()
