require('gnuplot')
require('nn')
grad = require 'autograd'
grad.optimize(true)

local n_time_slots = 10
-- local n_units_consumers = {20, 30, 25, 15}
local consumer_units_min = 15
local consumer_units_max = 35
local n_hid = 20
-- timeslots = torch.Tensor{0,1, 4,6,2,1,1,4,1,0}

function random_slots_distribution(n_time_slots, n_units)
    local rs = {}
    for i = 1, n_time_slots do rs[i] = torch.random(0,20) end
    rs = torch.Tensor(rs)
    rs = rs / torch.sum(rs)
    return torch.round(rs * n_units)
end

local dso_min_units = 1
local dso_max_units = 4

do
    local Consumer = torch.class('Consumer')

    function Consumer:__init(ind, n_units, total_timeslots)
        self.ind = ind
        self.n_units = n_units
        local mlp = nn.Sequential()
        mlp:add(nn.Linear(total_timeslots, n_hid))
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(n_hid, total_timeslots))
        mlp:add(nn.Abs())
        -- mlp:add(nn.SoftMax())
        self.modelf, self.params = grad.functionalize(mlp)
        self.neuralNet = function(params, prices)
           local units_distribution = self.modelf(params, prices)
           -- local units_distribution = torch.round(distribution * n_units)
           local to_pay = prices:view(-1) * units_distribution:view(-1)
           local loss = to_pay
           local units_diff = torch.sum(units_distribution) - self.n_units
           loss =  loss + units_diff^2
           return loss
        end
        self.grad = grad(self.neuralNet, {optimize = true})
    end
end

function Consumer:calculate_preferences(input)
    local distribution = self.modelf(self.params, input)
    -- return torch.round(distribution * n_units)
    return distribution
end

function Consumer:improve(prices)
    grads, loss = self.grad(self.params, prices)
    if g_opts.verbose then
        print('consumer loss', loss)
    end
    for i = 1, #self.params do
        self.params[i] = self.params[i] - g_opts.learning_rate * grads[i]
    end
end

do
    local DSO = torch.class('DSO')

    function DSO:__init()
        local mlp = nn.Sequential()
        mlp:add(nn.Linear(n_time_slots, n_hid)) -- Each DSO only receives usage of its own energy
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(n_hid, n_time_slots))
        mlp:add(nn.Abs())
        self.modelf, self.params = grad.functionalize(mlp)

        self.neuralNet = function(params, usage)
           local prices = self.modelf(params, usage)
           -- local loss = -torch.sum(torch.pow(torch.abs(usage - dso_min_units) + torch.abs(usage - dso_max_units), 2))
           local too_high = torch.sum(torch.cmax(usage, dso_max_units) - dso_max_units)
           local too_low = -torch.sum(torch.cmin(usage, dso_min_units) - dso_min_units)
           local loss = (too_low + too_high)^2
           local profit = usage:view(-1) * prices:view(-1)
           loss = loss - 0.05 * profit
           return loss
        end
        self.grad = grad(self.neuralNet, {optimize = true})
    end
end

function DSO:calculate_preferences(input)
    return self.modelf(self.params, input)
end

function DSO:improve(usage)
    grads, loss = self.grad(self.params, usage)
    for i = 1, #self.params do
        self.params[i] = self.params[i] - g_opts.learning_rate * grads[i]
    end
end

do
    local Game = torch.class('Game')

    function Game:__init(n_consumers, n_dsos)
        self.n_consumers = n_consumers
        self.n_dsos = n_dsos
        self.consumers = {}
        self.total_timeslots = n_time_slots * n_dsos
        self.consumers_usages = torch.Tensor(n_consumers, self.total_timeslots)
        for i=1,n_consumers do
            local n_units = torch.random(consumer_units_min, consumer_units_max)
            self.consumers[i] = Consumer(i, n_units, self.total_timeslots)
            self.consumers[i].usage = random_slots_distribution(self.total_timeslots, n_units)
            self.consumers_usages[i] = self.consumers[i].usage
        end
        local agg_usage = torch.sum(self.consumers_usages, 1)
        local splitted_by_dso = agg_usage:split(n_time_slots, 2)
        self.dsos = {}
        for i=1,n_dsos do
            self.dsos[i] = DSO()
            self.dsos[i].prices = self.dsos[i]:calculate_preferences(splitted_by_dso[i])
        end
        self.paid_by_consumers = {}
        self.total_usage_by_consumer = {}
        for i = 1, #self.consumers do
            self.paid_by_consumers[i] = {}
            self.total_usage_by_consumer[i] = {}
        end
        self.dso_profits = {}
        self.prices = {}
        for i = 1, #self.dsos do
            self.dso_profits[i] = {}
            self.prices[i] = {}
        end
    end
end

function Game:reset_results()
    for i = 1, #self.consumers do self.paid_by_consumers[i] = {} end
end

function Game:all_dso_prices()
    local all_dso_prices = {}
    for i, dso in pairs(self.dsos) do
        all_dso_prices[#all_dso_prices + 1] = dso.prices
    end
    return torch.cat(all_dso_prices)
end

function Game:play()
    local consumers_usage = torch.Tensor(self.n_consumers, self.total_timeslots)
    -- Calculate how much units the consumers will use
    local all_dso_prices = self:all_dso_prices()
    for i, consumer in pairs(self.consumers) do
        consumer.usage = consumer:calculate_preferences(all_dso_prices)
        consumer.total_usage = consumer.usage:sum()
        if g_opts.verbose then
            print('C' .. i .. ' usage', consumer.usage)
            print('C' .. i .. ' total units', torch.sum(consumer.usage))
        end
        consumers_usage[i] = consumer.usage
        consumer.paid = consumer.usage:view(-1) * all_dso_prices:view(-1)
    end
    local agg_usage = torch.sum(consumers_usage, 1)
    local splitted_by_dso = agg_usage:chunk(self.n_dsos, 2)
    if g_opts.verbose then
        print('aggregated usage', agg_usage)
    end
    for i,dso in pairs(self.dsos) do
        dso.profit = splitted_by_dso[i]:view(-1) * dso.prices:view(-1)
         if g_opts.verbose then
            print('DSO' .. i .. ' profit', dso.profit)
        end
        -- Calculate how much the DSO will charge next
        dso.prices = dso:calculate_preferences(splitted_by_dso[i])
        if g_opts.verbose then
            print('new DSO' .. i .. ' prices', dso.prices)
        end
    end
    all_dso_prices = self:all_dso_prices()
    for _, consumer in pairs(self.consumers) do
        consumer:improve(all_dso_prices)
    end
    for i,dso in pairs(self.dsos) do
        dso:improve(splitted_by_dso[i])
    end
end

function Game:play_multiple(n)
    for i = 1,n do
        xlua.progress(i, n)
        g:play()
        for i, consumer in pairs(self.consumers) do
            self.paid_by_consumers[i][#self.paid_by_consumers[i] + 1] = consumer.paid
            self.total_usage_by_consumer[i][#self.total_usage_by_consumer[i] + 1] = consumer.total_usage
        end
        for i,dso in pairs(self.dsos) do
            self.dso_profits[i][#self.dso_profits[i] + 1] = dso.profit
            self.prices[i][#self.prices[i] + 1] = dso.prices[1][1]
        end
    end
    local toplot = {}
    for i, consumer in pairs(self.consumers) do
        toplot[i] = {'Consumer' .. i, torch.Tensor(self.paid_by_consumers[i])}
    end
    gnuplot.figure()
    gnuplot.plot(toplot)
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Amount paid')
    gnuplot.plotflush()
    -- Total usage by consumer
    local toplot = {}
    for i, consumer in pairs(self.consumers) do
        toplot[i] = {'Consumer' .. i, torch.Tensor(self.total_usage_by_consumer[i])}
    end
    gnuplot.figure()
    gnuplot.plot(toplot)
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Total usage')
    gnuplot.plotflush()
    -- Profit of DSOs
    gnuplot.figure()
    toplot = {}
    for i,dso in pairs(self.dsos) do
        toplot[i] = {'DSO' .. i, torch.Tensor(self.dso_profits[i])}
    end
    gnuplot.plot(toplot)
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Profit')
    gnuplot.plotflush()
    -- Prices of DSOs
    gnuplot.figure()
    toplot = {}
    for i,dso in pairs(self.dsos) do
        toplot[i] = {'DSO' .. i, torch.Tensor(self.prices[i])}
    end
    gnuplot.plot(toplot)
    gnuplot.xlabel('Games')
    gnuplot.ylabel('Price for timeslot 1')
    gnuplot.plotflush()
end

local cmd = torch.CmdLine()
cmd:option('--n_consumers', 2, 'Amount of consumers')
cmd:option('--n_dsos', 1, 'Amount of DSOs')
cmd:option('--n_actions', 2, 'Amount of actions')
cmd:option('--learning_rate', 0.001, 'Learning rate')
cmd:option('--n_games', '100', 'Amount of games to play')
cmd:option('--verbose', false, 'Print prices and usages')

g_opts = cmd:parse(arg or {})

g = Game(g_opts.n_consumers, g_opts.n_dsos)
g:play_multiple(g_opts.n_games)
