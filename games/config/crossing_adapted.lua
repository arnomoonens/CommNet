-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

if not g_opts then g_opts = {} end

g_opts.multigames = {}

local mapH = torch.Tensor{7,7,7,7,1}
local mapW = torch.Tensor{7,7,7,7,1}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = 0
sso.costs.step = 0
sso.costs.pass = 0
sso.costs.collision = 10
sso.costs.wait = 0.01
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.add_block = false
sso.visibility = g_opts.visibility
sso.nagents = g_opts.nagents

sso.add_rate = 1.0
sso.max_agents = 2

-------------------------------------------------------
local CrossingAdaptedRangeOpts = {}
CrossingAdaptedRangeOpts.mapH = mapH:clone()
CrossingAdaptedRangeOpts.mapW = mapW:clone()

local CrossingAdaptedStaticOpts = {}
for i,j in pairs(sso) do CrossingAdaptedStaticOpts[i] = j end

local CrossingAdaptedOpts ={}
CrossingAdaptedOpts.RangeOpts = CrossingAdaptedRangeOpts
CrossingAdaptedOpts.StaticOpts = CrossingAdaptedStaticOpts

g_opts.multigames.CrossingAdapted = CrossingAdaptedOpts

return g_opts
