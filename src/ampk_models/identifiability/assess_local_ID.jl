using StructuralIdentifiability

# file_list = ["../odes/ampk_MA_double_mech.jl", "../odes/ampk_MA_single_mech.jl", "../odes/ampk_MM_double_mech.jl", "../odes/ampk_MM_single_mech.jl", "../odes/ampk_qss_single.jl", "../odes/ampk_qss2_single.jl"]
# model_list = ["MA_double", "MA_single", "MM_double", "MM_single", "qss_single", "qss2_single"]

file_list = ["../odes/ampk_qss_single.jl", "../odes/ampk_qss2_single.jl"]
model_list = ["qss_single", "qss2_single"]

for (file, model) in zip(file_list, model_list)
    println("Running: $model...")
    include(file)

    local_id = assess_local_identifiability(ode, 0.99)

    # save to file for this model
    # create file to save results
    file = open("../identifiability/local_ID_$model.txt", "w")
    println(file, "Locally Identifiable parameters:")
    for (key, value) in local_id
        if value == 1
            print(file, "$key, ")
        end
    end
    println(file, "")
    println(file, "Nonidentifiable parameters:")
    for (key, value) in local_id
        if value == 0
            print(file, "$key, ")
        end
    end
    close(file)
end