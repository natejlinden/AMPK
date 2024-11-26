using StructuralIdentifiability

# Define
# use the concentration of the product as the output (observable)
hill = @ODEmodel(
    x1'(t) = -(Vmax*x1(t)*x1(t))/(Km*Km + x1(t)*x1(t)), # S
    x2'(t) = (Vmax*x1(t)*x1(t))/(Km*Km + x1(t)*x1(t)), # P
    y1(t) = x2(t)
)

# Assess local identifiability with all parameters free
local_id_hill = assess_local_identifiability(hill)

# # assess global identifiability
id_hill = assess_identifiability(hill)

# assess global identifiability, assume that ICs are known
id_hill_known_ic = assess_identifiability(hill, known_ic = [x1,x2])

# write everything to a file 
fname = "../../results/MA_v_MM/results_hill.txt"
if isfile(fname)
    rm(fname)
end

file = open(fname, "w")
println(file, "MICHAELIS-MENTEN MODEL:")
println(file, "")
println(file, "LOCAL ID:")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id_hill
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id_hill
    if value == 0
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
println(file, "GLOBAL ID:")
println(file, "Globally:")
for (key, value) in id_hill
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Locally:")
for (key, value) in id_hill
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable:")
for (key, value) in id_hill
    if value == :nonidentifiable
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
println(file, "GLOBAL ID, known IC:")
println(file, "Globally:")
for (key, value) in id_hill_known_ic
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Locally:")
for (key, value) in id_hill_known_ic
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable:")
for (key, value) in id_hill_known_ic
    if value == :nonidentifiable
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")


close(file)