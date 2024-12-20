using StructuralIdentifiability

# Define model
# use the concentration of the product as the output (observable)
MA = @ODEmodel(
    x1'(t) = -kf_1*x1(t)*x2(t) + kr_1*x3(t) + kcat_2*x6(t), # S
    x2'(t) = -kf_1*x1(t)*x2(t) + kr_1*x3(t) + kcat_1*x3(t), # E_1
    x3'(t) =  kf_1*x1(t)*x2(t) - kr_1*x3(t) - kcat_1*x3(t), # E_1S
    x4'(t) =  kcat_1*x3(t), # P_1
    x5'(t) = -kf_2*x4(t)*x5(t) + kr_2*x6(t), # E_2
    x6'(t) = kf_2*x4(t)*x5(t) - kr_2*x6(t) - kcat_2*x6(t), # E_2P_1
    y1(t) = x4(t)+x6(t) #  total P_1
)

# Assess local identifiability with all parameters free
local_id_MA = assess_local_identifiability(MA)

# # # assess global identifiability
# id_MA = assess_identifiability(MA)

# # assess global identifiability, assume that ICs are known
# id_MA_known_ic = assess_identifiability(MA, known_ic = [x1,x2,x3,x4,x5,x6])

# write everything to a file 
fname = "../../results/MA_v_MM/results_MA_MA.txt"
if isfile(fname)
    rm(fname)
end

file = open(fname, "w")
println(file, "MASS ACTION-MASS ACTION MODEL:")
println(file, "")
println(file, "LOCAL ID:")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id_MA
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id_MA
    if value == 0
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
# println(file, "GLOBAL ID:")
# println(file, "Globally:")
# for (key, value) in id_MA
#     if value == :globally
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "Locally:")
# for (key, value) in id_MA
#     if value == :globally
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "Nonidentifiable:")
# for (key, value) in id_MA
#     if value == :nonidentifiable
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "")
# println(file, "")
# println(file, "GLOBAL ID, known IC:")
# println(file, "Globally:")
# for (key, value) in id_MA_known_ic
#     if value == :globally
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "Locally:")
# for (key, value) in id_MA_known_ic
#     if value == :globally
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "Nonidentifiable:")
# for (key, value) in id_MA_known_ic
#     if value == :nonidentifiable
#         print(file, "$key, ")
#     end
# end
# println(file, "")
# println(file, "")
close(file)