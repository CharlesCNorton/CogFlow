import os
import sys
import cv2
import torch
import random
import string
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import colorama
from colorama import Fore, init, Style

# Try importing local LLM library (ollama)
try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

init(autoreset=True)

################################################################################
#                             GLOBAL CONSTANTS
################################################################################

# Default CogVideoX generation parameters
COGVIDEO_WIDTH = 1280
COGVIDEO_HEIGHT = 720
COGVIDEO_NUM_FRAMES = 81   # e.g., ~5 seconds at 16 fps
COGVIDEO_INFER_STEPS = 50
COGVIDEO_FPS = 16

################################################################################
#                          UTILITY / HELPER FUNCTIONS
################################################################################

def pick_folder(title="Select Folder") -> str:
    """
    Opens a Tkinter folder selection dialog.

    Args:
        title (str): Title to display on the dialog.

    Returns:
        str: Directory path selected by the user (empty if none selected).
    """
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

def generate_random_string(length=8) -> str:
    """
    Generates a random alphanumeric string.

    Args:
        length (int): Desired length of the random string.

    Returns:
        str: Randomly generated string of specified length.
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def frames_to_mp4(frames, mp4_path: str, fps: int) -> None:
    """
    Converts a list of frames (NumPy arrays or GPU Tensors) to an MP4 file via OpenCV.

    Args:
        frames: A list of frames (in HWC format, with shape [..., 3] for color).
        mp4_path (str): Destination path for the MP4.
        fps (int): Frames per second for the output video.
    """
    if not frames:
        print(f"{Fore.RED}No frames provided. Unable to export video.")
        return

    first_frame = frames[0]
    if hasattr(first_frame, "cpu"):
        first_frame = first_frame.cpu().numpy().astype("uint8")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for idx, frm in enumerate(frames):
        if hasattr(frm, "cpu"):
            frm = frm.cpu().numpy().astype("uint8")
        if frm.shape[-1] == 3:  # Convert RGB to BGR for OpenCV
            frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        writer.write(frm)

    writer.release()
    print(f"{Fore.GREEN}Video saved to {mp4_path}")

def video_to_frames(mp4_path: str, frames_dir: str) -> int:
    """
    Extracts frames from an MP4 file, saving each frame as a .png file.

    Args:
        mp4_path (str): Path to the video file.
        frames_dir (str): Output directory to store .png frames.

    Returns:
        int: Number of frames extracted.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = None
    idx = 0

    try:
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            idx += 1
        print(f"{Fore.GREEN}Extracted {idx} frames from {mp4_path} -> {frames_dir}")
    except Exception as e:
        print(f"{Fore.RED}Error extracting frames: {e}")
    finally:
        if cap:
            cap.release()
    return idx

################################################################################
#                            COGFLOW CLASS
################################################################################

class CogFlowVideo:
    """
    A class that manages:
      - CogVideoX pipeline loading and short video generation
      - Florence-2 model loading and detection on extracted frames
      - YOLO annotations for each frame
      - YAML dataset creation

    Attributes:
        device (torch.device): CUDA if available, else CPU
        cogvideo_pipe (DiffusionPipeline): Diffusers pipeline for CogVideoX
        florence_model (AutoModelForCausalLM): Florence-2 model
        florence_processor (AutoProcessor): Processor for Florence-2
        cogvideo_dir (str): Directory containing CogVideoX model
        florence_dir (str): Directory containing Florence-2 model
        dataset_dir (str): Root directory for dataset (images/ labels/ etc.)
        output_mp4_dir (str): Folder to store raw MP4 files
        detection_classes (list[str]): Classes we want to detect & label in YOLO
        use_ollama (bool): Whether to use local LLM (ollama) for prompt generation
        ollama_model_name (str): Name of the local LLM model (e.g. "llama2.7b")
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cogvideo_pipe = None
        self.florence_model = None
        self.florence_processor = None

        self.cogvideo_dir = ""
        self.florence_dir = ""
        self.dataset_dir = ""
        self.output_mp4_dir = ""
        self.detection_classes = []

        self.use_ollama = True
        self.ollama_model_name = "llama2.7b"
        print(f"{Fore.CYAN}Initialized CogFlowVideo on device={self.device}")

    ############################################################################
    #                         MODEL LOADING METHODS
    ############################################################################

    def load_cogvideo_pipeline(self) -> None:
        """
        Loads the CogVideoX Diffusers pipeline from self.cogvideo_dir.
        """
        if not self.cogvideo_dir:
            print(f"{Fore.RED}No CogVideoX directory set.")
            return

        print(f"{Fore.CYAN}Loading CogVideoX from {self.cogvideo_dir} ...")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                self.cogvideo_dir,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            self.cogvideo_pipe = pipe
            print(f"{Fore.GREEN}CogVideoX loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}Error loading CogVideoX pipeline: {e}")
            self.cogvideo_pipe = None

    def load_florence_model(self) -> None:
        """
        Loads the Florence-2 model from self.florence_dir.
        """
        if not self.florence_dir:
            print(f"{Fore.RED}No Florence-2 directory set.")
            return

        print(f"{Fore.CYAN}Loading Florence-2 from {self.florence_dir} ...")
        try:
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                self.florence_dir, trust_remote_code=True
            ).eval().to(self.device).half()

            self.florence_processor = AutoProcessor.from_pretrained(
                self.florence_dir, trust_remote_code=True
            )
            print(f"{Fore.GREEN}Florence-2 model loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}Error loading Florence-2: {e}")
            self.florence_model = None
            self.florence_processor = None

    ############################################################################
    #                      LOCAL LLM PROMPT GENERATION (OLLAMA)
    ############################################################################

    def generate_local_prompt(self) -> str:
        """
        Generates a text prompt using the local LLM (ollama), if enabled and installed.

        Returns:
            str: The generated prompt (or user-provided fallback).
        """
        if not self.use_ollama or not OLLAMA_AVAILABLE:
            return input("Enter a text prompt manually: ")

        ans = input("Use local LLM (ollama) to generate a prompt? (y/n): ").strip().lower()
        if ans == 'y':
            try:
                user_req = "Create a short text prompt describing a 5-second, 720p comedic video scene."
                resp = chat(model=self.ollama_model_name, messages=[{"role": "user", "content": user_req}], stream=False)
                prompt_text = resp["message"]["content"].strip()
                print(f"{Fore.GREEN}LLM-based prompt: {prompt_text}")
                return prompt_text
            except Exception as e:
                print(f"{Fore.RED}Ollama error: {e}")
        return input("Enter a text prompt manually: ")

    ############################################################################
    #                       COGVIDEO GENERATION & DETECTION
    ############################################################################

    def create_cogvideo_clip(self, prompt: str) -> str:
        """
        Generates a short MP4 using CogVideoX. Stores it in self.output_mp4_dir.

        Args:
            prompt (str): The text prompt for CogVideoX.

        Returns:
            str: The path to the generated MP4 (or None if error).
        """
        if not self.cogvideo_pipe:
            print(f"{Fore.RED}CogVideoX pipeline not loaded.")
            return None

        if not self.output_mp4_dir:
            print(f"{Fore.RED}No output MP4 directory set.")
            return None

        vid_filename = f"generated_{generate_random_string()}.mp4"
        outpath = os.path.join(self.output_mp4_dir, vid_filename)

        print(f"{Fore.CYAN}Generating video for prompt: '{prompt}'")
        try:
            result = self.cogvideo_pipe(
                prompt=prompt,
                width=COGVIDEO_WIDTH,
                height=COGVIDEO_HEIGHT,
                num_frames=COGVIDEO_NUM_FRAMES,
                num_videos_per_prompt=1,
                guidance_scale=6.0,
                num_inference_steps=COGVIDEO_INFER_STEPS,
                generator=torch.Generator(self.device).manual_seed(42)
            )
            frames = result.frames[0]
            frames_to_mp4(frames, outpath, COGVIDEO_FPS)
            return outpath
        except Exception as e:
            print(f"{Fore.RED}Video generation error: {e}")
            return None

    def run_florence_detection(self, pil_image: Image.Image) -> dict:
        """
        Runs Florence-2 object detection on a single PIL image.

        Args:
            pil_image (Image.Image): Frame as a PIL image (RGB).

        Returns:
            dict: The parsed detection results, e.g. {"<OD>": {"labels": [...], "bboxes": [...]}}.
                  Returns None on error or if not loaded.
        """
        if not self.florence_model or not self.florence_processor:
            print(f"{Fore.RED}Florence-2 not loaded.")
            return None
        print(f"{Fore.CYAN}Running Florence detection...")

        try:
            task_prompt = "<OD>"
            inputs = self.florence_processor(text=task_prompt, images=pil_image, return_tensors="pt").to(self.device)
            original_size = pil_image.size

            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.half()

            with torch.amp.autocast("cuda"):
                gen_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1
                )
            raw_text = self.florence_processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = self.florence_processor.post_process_generation(
                raw_text, task=task_prompt, image_size=original_size
            )
            return parsed
        except Exception as e:
            print(f"{Fore.RED}Error in detection: {e}")
            return None

    def save_yolo_annotations(self, frame_path: str, parsed_res: dict, image_size: tuple) -> None:
        """
        Saves detection results in YOLO format (one .txt per frame).

        Args:
            frame_path (str): The .png file path of this frame.
            parsed_res (dict): The detection results from Florence.
            image_size (tuple): (width, height) of the image.
        """
        if "<OD>" not in parsed_res:
            return

        txt_path = frame_path.replace(".png", ".txt")
        bboxes = parsed_res["<OD>"].get("bboxes", [])
        labels = parsed_res["<OD>"].get("labels", [])
        w, h = image_size

        try:
            with open(txt_path, "w") as f:
                for bbox, label in zip(bboxes, labels):
                    if label.lower() in [cls.lower() for cls in self.detection_classes]:
                        cls_id = self.detection_classes.index(label.lower())
                        x1, y1, x2, y2 = bbox
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            print(f"{Fore.GREEN}Saved YOLO annotation: {txt_path}")
        except Exception as e:
            print(f"{Fore.RED}Annotation save error: {e}")

    def detect_frames_and_annotate(self, frames_dir: str) -> bool:
        """
        Runs detection on all frames in frames_dir. If no frame has a target class,
        returns False; otherwise True. Creates YOLO .txt for each detected frame.

        Args:
            frames_dir (str): Directory containing .png frames.

        Returns:
            bool: True if we found at least one target class in any frame; else False.
        """
        keep_video = False
        frames_list = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

        for frm_name in frames_list:
            frame_path = os.path.join(frames_dir, frm_name)
            pil_img = Image.open(frame_path).convert("RGB")
            res = self.run_florence_detection(pil_img)
            if res and "<OD>" in res:
                labels = res["<OD>"].get("labels", [])
                # If any label matches detection_classes
                for lbl in labels:
                    if lbl.lower() in [cls.lower() for cls in self.detection_classes]:
                        keep_video = True
                self.save_yolo_annotations(frame_path, res, pil_img.size)

        return keep_video

    def generate_and_detect_video(self) -> None:
        """
        Creates a short video (using either local LLM prompt or manual input),
        extracts frames, runs Florence detection, and removes the MP4 if no target classes found.
        """
        if not self.cogvideo_pipe:
            print(f"{Fore.RED}CogVideoX not loaded.")
            return
        if not self.output_mp4_dir:
            print(f"{Fore.RED}No output MP4 directory set.")
            return
        if not self.detection_classes:
            print(f"{Fore.RED}No detection classes set.")
            return

        prompt = self.generate_local_prompt()
        clip_path = self.create_cogvideo_clip(prompt)
        if not clip_path:
            print(f"{Fore.RED}Video generation failed.")
            return

        # Extract frames
        frames_subdir = f"frames_{generate_random_string()}"
        frames_dir = os.path.join(self.output_mp4_dir, frames_subdir)
        count_frames = video_to_frames(clip_path, frames_dir)
        if count_frames < 1:
            print(f"{Fore.RED}No frames extracted; removing {clip_path}")
            try:
                os.remove(clip_path)
            except:
                pass
            shutil.rmtree(frames_dir, ignore_errors=True)
            return

        keep = self.detect_frames_and_annotate(frames_dir)
        if not keep:
            print(f"{Fore.YELLOW}No target classes found in entire clip. Removing {clip_path}")
            try:
                os.remove(clip_path)
            except:
                pass
            shutil.rmtree(frames_dir, ignore_errors=True)
        else:
            print(f"{Fore.GREEN}Video retained: {clip_path}")
            print(f"{Fore.GREEN}Annotated frames left in {frames_dir}")

    ############################################################################
    #                    YAML & DATASET STRUCTURE FOR YOLO
    ############################################################################

    def create_dataset_structure_and_yaml(self) -> None:
        """
        Creates the typical YOLO-style folders under dataset_dir/images and dataset_dir/labels,
        plus a data.yaml referencing the detection classes.
        """
        if not self.dataset_dir:
            print(f"{Fore.RED}No dataset directory set.")
            return
        if not self.detection_classes:
            print(f"{Fore.RED}No detection classes specified.")
            return

        splits = ["train", "val", "eval"]
        for sp in splits:
            os.makedirs(os.path.join(self.dataset_dir, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "labels", sp), exist_ok=True)
        print(f"{Fore.GREEN}Created YOLO subfolders under {self.dataset_dir}")

        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        try:
            with open(yaml_path, "w") as f:
                f.write(
                    f"path: {self.dataset_dir}\n"
                    f"train: images/train\n"
                    f"val: images/val\n"
                    f"eval: images/eval\n"
                    f"nc: {len(self.detection_classes)}\n"
                    f"names: {self.detection_classes}\n"
                )
            print(f"{Fore.GREEN}Created data.yaml at {yaml_path}")
        except Exception as e:
            print(f"{Fore.RED}Error creating data.yaml: {e}")

################################################################################
#                                   MENU
################################################################################

def main_menu():
    """
    Provides an interactive menu to configure paths, load models,
    set detection classes, run generation/detection, and create dataset structure.
    """
    colorama.init(autoreset=True)
    obj = CogFlowVideo()

    while True:
        print(f"\n{Fore.BLUE}{Style.BRIGHT}{'='*40}")
        print(f"{Fore.CYAN}{Style.BRIGHT}          CogFlow Pipeline Menu          {Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*40}{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}{Style.BRIGHT}[I] Initialization & Paths{Style.RESET_ALL}")
        print("1. Select CogVideoX Model Directory")
        print("2. Select Florence-2 Model Directory")
        print("3. Select Output MP4 Directory")
        print("4. Select Dataset Directory (for YOLO data)")

        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[C] Classes & Models{Style.RESET_ALL}")
        print("5. Set Detection Classes (comma-separated)")
        print("6. Load CogVideoX Pipeline")
        print("7. Load Florence-2 Model")

        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[L] Local LLM Settings{Style.RESET_ALL}")
        print(f"8. Toggle Use Ollama (current={obj.use_ollama})")
        print(f"9. Set Ollama Model Name (current={obj.ollama_model_name})")

        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[R] Run & Generate{Style.RESET_ALL}")
        print("10. Generate Short Video & Annotate (CogVideoX -> frames -> Florence-2)")

        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[D] Dataset & Exit{Style.RESET_ALL}")
        print("11. Create YOLO dataset structure + data.yaml")
        print("12. Exit")

        choice = input(f"{Fore.YELLOW}Enter choice: {Style.RESET_ALL}").strip()
        if choice == "1":
            d = pick_folder("Select CogVideoX Directory")
            if d:
                obj.cogvideo_dir = d
                print(f"CogVideoX dir set: {obj.cogvideo_dir}")
        elif choice == "2":
            d = pick_folder("Select Florence-2 Directory")
            if d:
                obj.florence_dir = d
                print(f"Florence-2 dir set: {obj.florence_dir}")
        elif choice == "3":
            d = pick_folder("Select Output MP4 Directory")
            if d:
                obj.output_mp4_dir = d
                print(f"Output MP4 dir set: {obj.output_mp4_dir}")
        elif choice == "4":
            d = pick_folder("Select Dataset Directory (YOLO data)")
            if d:
                obj.dataset_dir = d
                print(f"Dataset dir set: {obj.dataset_dir}")
        elif choice == "5":
            val = input("Enter detection classes (comma-separated): ")
            if val.strip():
                obj.detection_classes = [c.strip() for c in val.split(",")]
                print(f"Detection classes={obj.detection_classes}")
        elif choice == "6":
            if not obj.cogvideo_dir:
                print(f"{Fore.RED}Set CogVideoX directory first.")
            else:
                obj.load_cogvideo_pipeline()
        elif choice == "7":
            if not obj.florence_dir:
                print(f"{Fore.RED}Set Florence-2 directory first.")
            else:
                obj.load_florence_model()
        elif choice == "8":
            ans = input(f"Use local LLM (ollama)? (y/n) current={obj.use_ollama}: ").strip().lower()
            obj.use_ollama = (ans == 'y')
            if obj.use_ollama and not OLLAMA_AVAILABLE:
                print(f"{Fore.RED}Ollama not installed. Setting use_ollama=False.")
                obj.use_ollama = False
        elif choice == "9":
            m = input(f"Enter Ollama model name (current={obj.ollama_model_name}): ").strip()
            if m:
                obj.ollama_model_name = m
        elif choice == "10":
            if not obj.cogvideo_pipe:
                print(f"{Fore.RED}CogVideo not loaded.")
            elif not obj.florence_model:
                print(f"{Fore.RED}Florence not loaded.")
            elif not obj.output_mp4_dir:
                print(f"{Fore.RED}No MP4 output directory set.")
            elif not obj.detection_classes:
                print(f"{Fore.RED}No detection classes set.")
            else:
                obj.generate_and_detect_video()
        elif choice == "11":
            obj.create_dataset_structure_and_yaml()
        elif choice == "12":
            print(f"{Fore.YELLOW}Exiting CogFlow.")
            break
        else:
            print(f"{Fore.RED}Invalid option. Try again.")

def main():
    main_menu()

if __name__ == "__main__":
    main()
