-- Change routes
-- Change route selection
-- Make them spawn at same time

local CrossingAdapted, parent = torch.class('CrossingAdapted', 'Traffic')

function CrossingAdapted:__init(opts, vocab)
    parent.__init(self, opts, vocab)
end

function CrossingAdapted:build_roads()
    -- build crossing
    assert(self.map.height % 2 == 1)
    assert(self.map.height == self.map.width)
    self.length = math.floor(self.map.height / 2)
    for y = 1, self.length do
        self:place_item({type = 'block'}, y, self.length)
        self:place_item({type = 'block'}, y, self.length + 2)
    end
    for y = self.length+2, self.map.height do
        self:place_item({type = 'block'}, y, self.length)
        self:place_item({type = 'block'}, y, self.length + 2)
    end
    for x = 1, self.length-1 do
        self:place_item({type = 'block'}, self.length, x)
        self:place_item({type = 'block'}, self.length + 2, x)
    end
    for x = self.length+3, self.map.width do
        self:place_item({type = 'block'}, self.length, x)
        self:place_item({type = 'block'}, self.length + 2, x)
    end
    table.insert(self.source_locs, {y = 1, x = self.length + 1, routes = {}})
    table.insert(self.source_locs, {y = self.length + 1, x = 1, routes = {}})
    table.insert(self.dest_locs, {y = self.map.height, x = self.length + 1})
    table.insert(self.dest_locs, {y = self.length + 1, x = self.map.width})

-- build routes
    local r

    -- From top to bottom
    r = {}
    for i = 1, self.map.height do
        table.insert(r, {y = i, x = self.length + 1})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[1].routes, #self.routes)

    -- From left to right
    r = {}
    for i = 1, self.map.width do
        table.insert(r, {y = self.length + 1, x = i})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[2].routes, #self.routes)

end