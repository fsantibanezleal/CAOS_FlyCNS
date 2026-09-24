# The two compound eyes: requirements

Written before the code (U2). Each requirement names the gate that fails when it is violated.

```
R-201  THE geometry measurement SHALL derive the fly's body axes (anterior, dorsal, left) from landmark neuropils of
       the release and SHALL return a right-handed frame.
       Gate: tests/test_eyes.py::test_body_axes_from_landmarks_are_right_handed

R-202  THE geometry measurement SHALL find, from the 3D centres of the medulla columns, which hexagonal offsets are
       nearest neighbours and which hex axis runs dorsal and which runs anterior in the medulla.
       Gate: tests/test_eyes.py::test_lattice_orientation_is_recovered_from_column_centres

R-203  THE eye model SHALL mirror the anterior-posterior axis between the medulla and the eye (the first optic
       chiasm) and SHALL keep the dorsal-ventral axis.
       Gate: tests/test_eyes.py::test_chiasm_mirrors_anterior_posterior_only

R-204  THE eye model SHALL place each eye's columns so that, along its equator, the most frontal column looks
       10 degrees into the opposite hemisphere and the most posterior looks 155 degrees back (Zhao et al. 2025).
       Gate: tests/test_eyes.py::test_equator_spans_the_measured_extent

R-205  THE eye model SHALL give neighbouring columns a nearly constant angular separation, and the left eye SHALL be
       the mirror image of the right when their lattices are the same.
       Gate: tests/test_eyes.py::test_neighbours_are_evenly_spaced_and_eyes_mirror

R-206  THE sampling matrix SHALL weight panorama pixels by a Gaussian acceptance of the configured half-width and by
       each pixel's solid angle, normalised so that each ommatidium's weights sum to one.
       Gate: tests/test_eyes.py::test_sampling_rows_sum_to_one_and_uniform_light_stays_uniform

R-207  WHEN a panorama holds a single bright spot, THE ommatidium looking closest to it SHALL respond most.
       Gate: tests/test_eyes.py::test_a_bright_spot_lights_the_ommatidium_that_looks_at_it

R-208  WHERE the compiled MaleCNS and its measured geometry are present, THE right eye SHALL hold 892 columns and the
       left 879, and in each eye the dorsal-rim columns (identified from their photoreceptor subtypes, not from
       geometry) SHALL sit higher on average than the pale and yellow columns.
       Gate: tests/test_eyes_data.py::test_malecns_eyes_match_the_release_and_the_measured_extent
```
