store = Ref(Dict{String,Tuple{Float64,Float64}}()) 
activated = Ref(false)
function perf!(name, t)
    if activated[]
        if haskey(store[], name)
            x = store[][name]
            store[][name] = (x[1]+t, x[2]+1.0)
        else
            store[][name] = (t, 1.0)
        end
    end
end


function activatePerf()
    activated[] = true
end

function compPerf()
    foreach(x->println("time $(x) : $(store[][x][1]/store[][x][2])"),keys(store[]))
end