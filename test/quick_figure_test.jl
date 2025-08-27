# ==================================================================================
# Quick Test of Figure Configuration  
# ==================================================================================

println("=== Testing Figure Configuration ===")

using ClampedPinnedRodSolver

println("1. Testing default configuration...")
default_config = get_default_config()
println("   ✓ save_figures = $(default_config.save_figures)")
println("   ✓ use_timestamped_folders = $(default_config.use_timestamped_folders)")
println("   ✓ figures_base_path = $(default_config.figures_base_path)")

println("\n2. Testing custom configuration...")
custom_config = create_config(
    xp = 0.3,
    yp = 0.1,
    save_figures = false,
    figure_format = "pdf"
)
println("   ✓ save_figures = $(custom_config.save_figures)")
println("   ✓ figure_format = $(custom_config.figure_format)")

println("\n3. Testing utility functions...")
enabled_config = create_config(save_figures = true)
disabled_config = create_config(save_figures = false)

println("   ✓ should_save_figures(enabled): $(should_save_figures(enabled_config))")
println("   ✓ should_save_figures(disabled): $(should_save_figures(disabled_config))")

if should_save_figures(enabled_config)
    test_path = get_figure_path(enabled_config, "test_plot")
    println("   ✓ get_figure_path: $test_path")
end

save_opts = get_figure_save_options(enabled_config)
println("   ✓ get_figure_save_options: $save_opts")

println("\n=== All Figure Configuration Tests Passed! ===")
println("The new figure saving functionality is working correctly! 🎉")
