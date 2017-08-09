require('gnuplot')
require('nn')
grad = require 'autograd'

local n_time_slots = 10
local n_units = 20
local n_hid = 20
-- timeslots = torch.Tensor{0,1, 4,6,2,1,1,4,1,0}

function random_slots_distribution()
    local rs = {}
    for i = 1, n_time_slots do rs[i] = torch.random(0,20) end
    rs = torch.Tensor(rs)
    rs = rs / torch.sum(rs)
    return torch.round(rs * n_units)
end

local min_units = 1
local max_units = 10

do
    local Consumer = torch.class('Consumer')

    function Consumer:__init()
        self.mlp = nn.Sequential()
        self.mlp:add(nn.Linear(n_time_slots, n_hid))
        self.mlp:add(nn.Tanh())
        self.mlp:add(nn.Linear(n_hid, n_time_slots))
        self.mlp:add(nn.Abs())
        -- self.mlp:add(nn.SoftMax())
        self.modelf, self.params = grad.functionalize(self.mlp)
        self.neuralNet = function(params, prices)
           local units_distribution = self.modelf(params, prices)
           -- local units_distribution = torch.round(distribution * n_units)
           local to_pay = prices:view(-1) * units_distribution:view(-1)
           local loss = 0
           if g_opts.verbose then
              print('to pay', to_pay.value)
          end
           local units_diff = torch.sum(units_distribution) - n_units
           if g_opts.verbose then
              print('units diff', units_diff.value)
          end
           loss = loss + units_diff
           return loss
        end
        self.grad = grad(self.neuralNet, {optimize = true})
    end
end

function Consumer:calculate_preferences(input)
    local distribution = self.mlp:forward(input)
    -- return torch.round(distribution * n_units)
    return distribution
end

function Consumer:improve(prices)
    grads, loss = self.grad(self.params, prices)
    for i = 1, #self.params do
        self.params[i]:add(-0.01, grads[i])
    end
end

do
    local DSO = torch.class('DSO')

    function DSO:__init()
        self.mlp = nn.Sequential()
        self.mlp:add(nn.Linear(n_time_slots, n_hid))
        self.mlp:add(nn.Tanh())
        self.mlp:add(nn.Linear(n_hid, n_time_slots))
        self.mlp:add(nn.Abs())
        self.modelf, self.params = grad.functionalize(self.mlp)

        self.neuralNet = function(params, usage)
           local prices = self.modelf(params, usage)
           -- local loss = -torch.sum(torch.pow(torch.abs(usage - min_units) + torch.abs(usage - max_units), 2))
           local too_high = torch.sum(torch.cmax(usage, max_units) - max_units)
           local too_low = -torch.sum(torch.cmin(usage, min_units) - min_units)
           local loss = (too_low + too_high)^2
           local profit = usage:view(-1) * prices:view(-1)
           loss = loss -  0.1 * profit
           return loss
        end
        self.grad = grad(self.neuralNet, {optimize = true})
    end
end

function DSO:calculate_preferences(input)
    return self.mlp:forward(input)
end

function DSO:improve(usage)
    grads, loss = self.grad(self.params, usage)
    for i = 1, #self.params do
        self.params[i]:add(-0.01, grads[i])
    end
    -- function feval( params )
    --     local outputs = self.mlp:forward(inputs)
    --     local grad = torch.Tensor(n_time_slots * n_hid * n_time_slots):zero()
    --     params, gradParams = mlp:getParameters()
    -- end
    -- optim.sgd()
end

do
    local Game = torch.class('Game')

    function Game:__init(n_consumers)
        self.n_consumers = n_consumers
        self.consumers = {}
        self.consumers_usages = torch.Tensor(n_consumers, n_time_slots)
        for i=1,n_consumers do
            self.consumers[i] = Consumer()
            self.consumers_usages[i] = random_slots_distribution()
        end
        local agg_usage = torch.sum(self.consumers_usages, 1)
        self.dso = DSO(agg_usage)
        self.results = {}
        for i = 1, #self.consumers do self.results[i] = {} end
        self.dso.prices = self.dso:calculate_preferences(agg_usage)
        self.dso.prices:reshape(n_time_slots)
        self.dso_profits = {}
    end
end

function Game:reset_results()
    for i = 1, #self.consumers do self.results[i] = {} end
end

function Game:play()
    local consumers_usage = torch.Tensor(self.n_consumers, n_time_slots)
    -- Calculate how much units the consumers will use
    for i, consumer in pairs(self.consumers) do
        local usage = consumer:calculate_preferences(self.dso.prices)
        if g_opts.verbose then
            print('Consumer usage', usage)
        end
        consumers_usage[i] = usage
        consumer.paid = usage:view(-1) * self.dso.prices:view(-1)
    end
    local agg_usage = torch.sum(consumers_usage, 1)
    self.dso.profit = agg_usage:view(-1) * self.dso.prices:view(-1)
     if g_opts.verbose then
        print('DSO profit', self.dso.profit)
    end
    -- Calculate how much the DSO will charge next
    self.dso.prices = self.dso:calculate_preferences(agg_usage)
    self.dso.prices:reshape(n_time_slots)
    if g_opts.verbose then
        print('new DSO prices', self.dso.prices)
    end
    -- print(self.dso.prices)
    -- for i, consumer in pairs(self.consumers) do
    --     consumer:improve(consumers_actions[i], r[i] + torch.uniform(-0.05,0.05))
    -- end
    self.dso:improve(agg_usage)
    for _, consumer in pairs(self.consumers) do
        consumer:improve(self.dso.prices)
    end
end

function Game:play_multiple(n)
    for i = 1,n do
        xlua.progress(i, n)
        g:play()
        for i, consumer in pairs(self.consumers) do
            self.results[i][#self.results[i] + 1] = consumer.paid
        end
        self.dso_profits[#self.dso_profits + 1] = self.dso.profit
    end
    local toplot = {}
    for i, consumer in pairs(self.consumers) do
        toplot[i] = {'Consumer' .. i, torch.Tensor(self.results[i])}
    end
    gnuplot.figure()
    gnuplot.plot(toplot)
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Amount paid')
    gnuplot.plotflush()
    gnuplot.figure()
    gnuplot.plot({'DSO', torch.Tensor(self.dso_profits)})
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Profit')
    gnuplot.plotflush()
end

local cmd = torch.CmdLine()
cmd:option('--n_consumers', 2, 'Number of consumers')
cmd:option('--n_actions', 2, 'Number of actions')
cmd:option('--n_games', '100', 'Number of games to play')
cmd:option('--verbose', false, 'Print prices and usages')

g_opts = cmd:parse(arg or {})

g = Game(g_opts.n_consumers)
g:play_multiple(g_opts.n_games)
