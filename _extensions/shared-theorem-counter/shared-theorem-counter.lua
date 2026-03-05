-- shared-theorem-counter.lua
-- Shares a single counter across all theorem-like environments within each chapter.
-- Must be placed AFTER 'quarto' in the filters list so it runs after
-- Quarto's built-in crossref processing.

local theorem_types = pandoc.List({
  "theorem", "lemma", "corollary", "proposition", "conjecture",
  "definition", "example", "exercise", "solution", "remark", "algorithm"
})

local shared_counter = 0
local chapter_prefix = nil
local id_to_number = {}

local function is_theorem_div(div)
  for _, cls in ipairs(div.classes) do
    if theorem_types:includes(cls) then
      return true
    end
  end
  return false
end

-- Parse a theorem title string like "Definition 2.1" or "Theorem 3 (Name)".
-- Returns (type_name, chapter_prefix, local_number) or nil.
local function parse_title(text)
  -- Normalize non-breaking spaces (UTF-8: \xc2\xa0) to regular spaces,
  -- since Quarto cross-references use &nbsp; between type name and number.
  text = text:gsub("\xc2\xa0", " ")
  -- With chapter numbering: "Type Ch.N" or "Type Ch.N (Name)"
  local type_name, ch, num = text:match("^(%a+)%s+(%d+%.)(%d+)")
  if type_name then
    return type_name, ch, num
  end
  -- Without chapter numbering: "Type N" or "Type N (Name)"
  type_name, num = text:match("^(%a+)%s+(%d+)")
  if type_name then
    return type_name, "", num
  end
  return nil, nil, nil
end

function Pandoc(doc)
  -- First pass: renumber all theorem-like environments with a shared counter
  local function renumber(blocks)
    for _, block in ipairs(blocks) do
      if block.t == "Div" and is_theorem_div(block) then
        shared_counter = shared_counter + 1

        if #block.content > 0 and block.content[1].t == "Para" then
          local para = block.content[1]
          for _, inline in ipairs(para.content) do
            if inline.t == "Span" and inline.classes:includes("theorem-title") then
              local text = pandoc.utils.stringify(inline)
              local type_name, ch, _ = parse_title(text)

              if type_name then
                -- Detect the chapter prefix from the first theorem encountered
                if chapter_prefix == nil then
                  chapter_prefix = ch
                end

                local new_num = (chapter_prefix or "") .. tostring(shared_counter)

                if block.identifier ~= "" then
                  id_to_number[block.identifier] = new_num
                end

                -- Preserve any name in parentheses, e.g. "(Fermat)"
                local name = text:match("(%(.+%))%s*$")
                local new_title = type_name .. " " .. new_num
                if name then
                  new_title = new_title .. " " .. name
                end

                inline.content = {pandoc.Strong({pandoc.Str(new_title)})}
              end
              break
            end
          end
        end
      end

      -- Recurse into Divs (sections, etc.)
      if block.t == "Div" then
        renumber(block.content)
      end
    end
  end

  renumber(doc.blocks)

  -- Second pass: update cross-reference links to use the new shared numbers.
  -- We emit raw HTML because Quarto's post-processing would otherwise
  -- overwrite the link text with its own (per-type) numbering.
  doc = doc:walk({
    Link = function(link)
      local target_id = link.target:match("^#(.+)$")
      if target_id and id_to_number[target_id] then
        local text = pandoc.utils.stringify(link)
        -- Normalize non-breaking spaces (UTF-8 \xc2\xa0) to regular spaces.
        text = text:gsub("\xc2\xa0", " ")
        -- Extract the type name (first word, ASCII letters only).
        local type_name = text:match("^([A-Za-z]+)")
        if type_name and theorem_types:includes(type_name:lower()) then
          return pandoc.RawInline("html",
            '<a href="#' .. target_id .. '">'
            .. type_name .. '&nbsp;' .. id_to_number[target_id]
            .. '</a>')
        end
      end
      return link
    end
  })

  return doc
end
