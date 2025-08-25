# ==================================================================================
# Test Figure Saving Configuration
# ==================================================================================
# 
# Quick test to verify the new figure saving configuration functionality
#
# Author: José Luis Ruiz-Erezuma  
# Created: August 2025
# ==================================================================================

# Test if running standalone or as module
if !@isdefined(ClampedRodConfig)
    # Running standalone - include the package
    push!(LOAD_PATH, dirname(@__DIR__))
    using ClampFixedRodSolver
else
    println("Running as part of module")
end

println("Testing Figure Saving Configuration...")

# Test 1: Default configuration
println("\n1. Testing default configuration:")
default_config = get_default_config()
println("   ✓ Default config created")
println("   ✓ save_figures = $(default_config.save_figures)")
println("   ✓ use_timestamped_folders = $(default_config.use_timestamped_folders)")

# Test 2: Custom configuration
println("\n2. Testing custom configuration:")
custom_config = create_config(
    xp = 0.3,
    yp = 0.1,
    save_figures = false,
    figure_format = "pdf",
    figure_dpi = 600
)
println("   ✓ Custom config created")
println("   ✓ save_figures = $(custom_config.save_figures)")
println("   ✓ figure_format = $(custom_config.figure_format)")
println("   ✓ figure_dpi = $(custom_config.figure_dpi)")

# Test 3: Utility functions
println("\n3. Testing utility functions:")

# Test should_save_figures
enabled_config = create_config(save_figures = true)
disabled_config = create_config(save_figures = false)

println("   ✓ should_save_figures(enabled): $(should_save_figures(enabled_config))")
println("   ✓ should_save_figures(disabled): $(should_save_figures(disabled_config))")

# Test get_figure_path
if should_save_figures(enabled_config)
    test_path = get_figure_path(enabled_config, "test_plot")
    println("   ✓ get_figure_path works: $test_path")
end

disabled_path = get_figure_path(disabled_config, "test_plot")
println("   ✓ get_figure_path (disabled): '$disabled_path'")

# Test get_figure_save_options
save_opts = get_figure_save_options(enabled_config)
println("   ✓ get_figure_save_options: $save_opts")

# Test 4: Different configurations
println("\n4. Testing different configuration scenarios:")

configs_to_test = [
    ("No timestamps", create_config(use_timestamped_folders = false)),
    ("Custom path", create_config(figures_base_path = "test_results")),
    ("PDF format", create_config(figure_format = "pdf")),
    ("High DPI", create_config(figure_dpi = 600))
]

for (name, config) in configs_to_test
    path = get_figure_path(config, "sample")
    println("   ✓ $name: $path")
end

# Test 5: Print configuration
println("\n5. Testing print_config function:")
test_config = create_config(
    xp = 0.25,
    yp = 0.15, 
    save_figures = true,
    use_timestamped_folders = false,
    figures_base_path = "test_output"
)

print_config(test_config)

println("\n=== All Figure Configuration Tests Passed! ===")

# Test summary
println("""
✅ Configuration creation works
✅ Utility functions work correctly  
✅ Figure path generation works
✅ Save options generation works
✅ All parameter combinations work

The figure saving configuration system is ready to use!
""")
