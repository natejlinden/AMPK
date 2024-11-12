using StructuralIdentifiability

# Define
# use the concentration of the product as the output (observable)
MM = @ODEmodel(
    x1'(t) = -(Vmax*x1(t))/(Km + x1(t)), # S
    x2'(t) = (Vmax*x1(t))/(Km + x1(t)), # P
    y1(t) = x2(t)/x1(t)
)

# Assess local identifiability with all parameters free
local_id_MM = assess_local_identifiability(MM)

# # assess global identifiability
id_MM = assess_identifiability(MM)

# assess global identifiability, assume that ICs are known
id_MM_known_ic = assess_identifiability(MM, known_ic = [x1,x2])

# write everything to a file 
fname = "./results_MM_ratio.txt"
if isfile(fname)
    rm(fname)
end

file = open(fname, "w")
println(file, "MICHAELIS-MENTEN MODEL:")
println(file, "")
println(file, "LOCAL ID:")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id_MM
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id_MM
    if value == 0
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
println(file, "GLOBAL ID:")
println(file, "Globally:")
for (key, value) in id_MM
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Locally:")
for (key, value) in id_MM
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable:")
for (key, value) in id_MM
    if value == :nonidentifiable
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
println(file, "GLOBAL ID, known IC:")
println(file, "Globally:")
for (key, value) in id_MM_known_ic
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Locally:")
for (key, value) in id_MM_known_ic
    if value == :globally
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable:")
for (key, value) in id_MM_known_ic
    if value == :nonidentifiable
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")


close(file)