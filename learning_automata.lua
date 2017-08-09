require('gnuplot')

local rewards = {}
rewards[2] = {
{{1, 0.5}, {0, 0}},
{{0, 0}, {0.5, 1}}
}

rewards[3] = {
{{{1, 0.5, 1}, {1, 0.5, 1}}, {{0, 0, 0}, {0, 0, 0}}},
{{{0, 0, 0}, {0, 0, 0}}, {{1, 0.5, 1}, {1, 0.5, 1}}}
}

local a = 0.1
local b = 0.005
local l = a / b

do
    local Agent = torch.class('Agent')

    function Agent:__init(probs)
        self.probs = torch.Tensor(probs)
    end
end

function Agent:select_action()
    return torch.multinomial(self.probs, 1)[1]
end

function Agent:improve(selected, reward)
    for i = 1, self.probs:size(1) do
        if i == selected then
            self.probs[i] = self.probs[i] + a * reward * (1 - self.probs[i]) - b * (1 - reward) * self.probs[i]
        else
            self.probs[i] = self.probs[i] - a * reward * self.probs[i] + b * (1 - reward) * (torch.pow(l - 1, -1) - self.probs[i])
        end
    end
end

do
    local Game = torch.class('Game')

    function Game:__init(n_agents, n_actions, rewards)
        probs = {}
        p = 1 / n_actions
        for i = 1, n_actions do
            probs[i] = p
        end
        self.agents = {}
        for i=1,n_agents do
            self.agents[i] = Agent(probs)
        end
        self.rewards = rewards
        self.results = {}
        for i = 1, #self.agents do self.results[i] = {} end
    end
end

function Game:reset_results()
    for i = 1, #self.agents do self.results[i] = {} end
end

function Game:play()
    local agents_actions = {}
    local r = self.rewards
    for i, agent in pairs(self.agents) do
        local action = agent:select_action()
        agents_actions[i] = action
        r = r[action]
    end
    for i, agent in pairs(self.agents) do
        agent:improve(agents_actions[i], r[i] + torch.uniform(-0.05,0.05))
    end
end

function Game:play_multiple(n)
    for i = 1,n do
        g:play()
        for i, agent in pairs(self.agents) do
            self.results[i][#self.results[i] + 1] = agent.probs[1]
        end
    end
    local toplot = {}
    for i, agent in pairs(self.agents) do
        toplot[i] = {'Agent' .. i, torch.Tensor(self.results[i])}
    end
    gnuplot.plot(toplot)
end

local cmd = torch.CmdLine()
cmd:option('--n_agents', 2, 'Number of agents')
cmd:option('--n_actions', 2, 'Number of actions')
cmd:option('--n_games', '100', 'Number of games to play')

g_opts = cmd:parse(arg or {})

g = Game(g_opts.n_agents, g_opts.n_actions, rewards[g_opts.n_agents])
g:play_multiple(g_opts.n_games)
