############ Run Inference with Cropping Algorithm ############################

# Load the input image.
num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

output_images = []

bar = display(progress(0, num_frames-1), display_id=True)

for frame_idx in range(num_frames):

    keypoints_with_scores = run_inference(
        movenet, image[frame_idx, :, :, :], crop_region, crop_size=[input_size, input_size])

    output_images.append(draw_prediction_on_image(
            image[frame_idx, :, :, :].numpy().astype(np.int32),
            keypoints_with_scores, crop_region=None,
            close_figure=True, output_image_height=300))

    crop_region = determine_crop_region( keypoints_with_scores, image_height, image_width)

    bar.update(progress(frame_idx, num_frames-1))

# Prepare gif visualization.
output = np.stack(output_images, axis=0)
to_gif(output, fps=10)
