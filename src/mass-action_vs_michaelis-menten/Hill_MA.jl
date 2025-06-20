using StructuralIdentifiability

# Define model
# use the concentration of the product as the output (observable)
MA = @ODEmodel(
    x1'(t) = -(Vmax_1*x1(t)*x1(t)/(Km_1*Km_1 + x1(t)*x1(t))) + kcat_2*x4(t),
    x2'(t) = (Vmax_1*x1(t)*x1(t)/(Km_1*Km_1 + x1(t)*x1(t))) - k_f_2*x2(t)*x3(t) + k_r_2*x4(t),
    x3'(t) = -k_f_2*x2(t)*x3(t) + k_r_2*x4(t) + kcat_2*x4(t),
    x4'(t) = k_f_2*x2(t)*x3(t) - k_r_2*x4(t) - kcat_2*x4(t),
    y1(t) = (x2(t)+x4(t)) # ratio of P_1 to S
)

# Assess local identifiability with all parameters free
local_id_MA = assess_local_identifiability(MA)

# # # assess global identifiability
# id_MA = assess_identifiability(MA)

# # assess global identifiability, assume that ICs are known
# id_MA_known_ic = assess_identifiability(MA, known_ic = [x1,x2,x3,x4])

# write everything to a file 
fname = "../../results/MA_v_MM/results_Hill_MA.txt"
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