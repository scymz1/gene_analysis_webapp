from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
import os
from django.conf import settings
import logging
import subprocess
from django.http import StreamingHttpResponse
from django.core.files.storage import default_storage
from datetime import datetime, timedelta
import traceback
import shutil
import time
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

def cleanup_old_folders(base_path, max_age_days=7):
    """
    Clean up folders older than max_age_days in the base_path directory.
    Folders are expected to be named with timestamp format: YYYYMMDD_HHMMSS
    """
    try:
        if not os.path.exists(base_path):
            logger.warning(f"Base path does not exist: {base_path}")
            return

        current_time = datetime.now()
        # List all directories in the base path
        for dirname in os.listdir(base_path):
            dir_path = os.path.join(base_path, dirname)
            
            # Skip if it's not a directory
            if not os.path.isdir(dir_path):
                continue
                
            try:
                # Parse the directory name as a timestamp
                dir_time = datetime.strptime(dirname, '%Y%m%d_%H%M%S')
                
                # Calculate the age of the directory
                age = current_time - dir_time
                
                # Remove if older than max_age_days
                if age > timedelta(days=max_age_days):
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed old directory: {dir_path}")
                    
            except ValueError:
                # Skip directories that don't match our timestamp format
                logger.warning(f"Skipping directory with invalid timestamp format: {dirname}")
                continue
            except Exception as e:
                logger.error(f"Error removing directory {dir_path}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in cleanup_old_folders: {str(e)}", exc_info=True)

@api_view(['GET'])
def download_result(request):
    try:
        output_dir = request.GET.get('output_dir')
        specific_file = request.GET.get('file')
        
        if not output_dir:
            return Response({'error': 'No output directory provided'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # If a specific file is requested, return that file
        if specific_file:
            file_path = os.path.join(output_dir, specific_file)
            if not os.path.exists(file_path):
                abs_path = os.path.join(os.path.dirname(__file__), file_path)
                if os.path.exists(abs_path):
                    file_path = abs_path
                else:
                    return Response({'error': f'File {specific_file} not found'}, 
                                  status=status.HTTP_404_NOT_FOUND)
            
            response = FileResponse(open(file_path, 'rb'))
            response['Content-Disposition'] = f'inline; filename="{specific_file}"'
            return response
        
        # Otherwise, compress the output directory and return it
        shutil.make_archive(output_dir, 'zip', output_dir)
        if not os.path.getsize(output_dir + '.zip'):
            return Response({'error': 'Result file not found'}, 
                          status=status.HTTP_404_NOT_FOUND)
        
        response = FileResponse(open(output_dir + '.zip', 'rb'))
        response['Content-Disposition'] = 'attachment; filename="result.zip"'
        return response
        
    except Exception as e:
        logger.error(f"Error downloading result: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error downloading result: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_cnnfold(fasta_path, output_dir):
    """Helper function to run CNNfold prediction"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the CNNfold directory path
        cnnfold_dir = os.path.join(
            os.path.dirname(__file__),
            "JiaLuModels/CNNfold/CNNfoldmodel"
        )
        
        # Set the Python path to include the CNNfold directory
        env = os.environ.copy()
        env['PYTHONPATH'] = cnnfold_dir + os.pathsep + env.get('PYTHONPATH', '')
        
        script_path = os.path.join(cnnfold_dir, "predict_only2.py")
        output_path = os.path.join(output_dir, 'result.db')
        
        # Change to the CNNfold directory before running the script
        current_dir = os.getcwd()
        os.chdir(cnnfold_dir)
        
        try:
            process = subprocess.Popen(
                f"python predict_only2.py --input {fasta_path} --output {output_path}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            print(process.stdout.read(), output_path)

            return Response({
                'message': 'CNNfold prediction completed', 
                'input_directory': fasta_path, 
                'output_directory': output_dir,
                'files_processed': 1
                }, status=status.HTTP_200_OK)
            
        finally:
            # Restore the original working directory
            os.chdir(current_dir)
        
    except Exception as e:
        logger.error(f"Error in predict_cnnfold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in CNNfold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_ufold(fasta_path, output_dir):
    """Helper function to run Ufold prediction"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the Ufold directory path
        ufold_dir = os.path.join(
            os.path.dirname(__file__),
            "JiaLuModels/UFold"
        )
        
        # Set the Python path and get current directory
        env = os.environ.copy()
        env['PYTHONPATH'] = ufold_dir + os.pathsep + env.get('PYTHONPATH', '')
        current_dir = os.getcwd()
        
        # Change to Ufold directory
        os.chdir(ufold_dir)

        try:
            process = subprocess.Popen(
                f"python run-ufold.py {fasta_path} {os.path.join(output_dir, 'result.db')} --nc True",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            output = process.stdout.read()
            print(f"UFold output: {output}")

            return Response({
                'message': 'Ufold prediction completed',
                'input_directory': fasta_path,
                'output_directory': output_dir,
                'files_processed': 1
            }, status=status.HTTP_200_OK)
            
        finally:
            # Restore the original working directory
            os.chdir(current_dir)
            
    except Exception as e:
        logger.error(f"Error in predict_ufold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in Ufold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_linearsampling(fasta_path, output_dir, use_shape=False, shape_path=None):
    """Helper function to run LinearSampling prediction with optional SHAPE data"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the LinearSampling directory path
        linear_sampling_dir = os.path.join(
            os.path.dirname(__file__),
            "JiaLuModels/LinearSampling"
        )
        
        # Set the Python path to include the LinearSampling directory
        env = os.environ.copy()
        env['PYTHONPATH'] = linear_sampling_dir + os.pathsep + env.get('PYTHONPATH', '')
        
        script_path = os.path.join(linear_sampling_dir, "linear-sampling.py")
        output_file = os.path.join(output_dir, "result.db")
        
        # Build the command based on whether SHAPE data is used
        if use_shape and shape_path:
            cmd = f"python {script_path} {fasta_path} {output_file} --shape"
            logger.info(f"Running LinearSampling with SHAPE data: {shape_path}")
        else:
            cmd = f"python {script_path} {fasta_path} {output_file}"
            logger.info("Running LinearSampling without SHAPE data")
        
        # Change to the LinearSampling directory before running the script
        current_dir = os.getcwd()
        os.chdir(linear_sampling_dir)
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Wait for process to complete
            process.wait()
            
            # Read output
            output = process.stdout.read()
            print(f"LinearSampling output: {output}")
            
            # Wait for output file to be generated with timeout
            # max_wait_time = 120  # Maximum wait time in seconds
            # wait_interval = 1    # Check interval in seconds
            # waited_time = 0
            
            # while not os.path.exists(output_file) and waited_time < max_wait_time:
            #     time.sleep(wait_interval)
            #     waited_time += wait_interval
            #     print(f"Waiting for output file... ({waited_time}s)")
            
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Output file not generated after {max_wait_time} seconds: {output_file}")
            
            # Ensure file is not empty
            if os.path.getsize(output_file) == 0:
                raise ValueError(f"Output file is empty: {output_file}")
                
            logger.info(f"LinearSampling prediction completed. Results saved to {output_file}")
            
            return Response({
                'message': 'LinearSampling prediction completed',
                'input_directory': fasta_path,
                'output_directory': output_dir,
                'files_processed': 1,
                'output_file': output_file
            }, status=status.HTTP_200_OK)
            
        finally:
            # Always restore the original working directory
            os.chdir(current_dir)
        
    except Exception as e:
        logger.error(f"Error in predict_linearsampling: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in LinearSampling prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def predict_redfold(fasta_path, output_dir):
    """Helper function to run RedFold prediction and save results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the RedFold directory path
        redfold_dir = os.path.join(os.path.dirname(__file__), "JiaLuModels/REDfold")
        fasta_dir = os.path.dirname(fasta_path)

        # Construct the output file path
        output_file = os.path.join(output_dir, "result.db")

        # Set the environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = redfold_dir + os.pathsep + env.get('PYTHONPATH', '')

        # Run RedFold with subprocess
        process = subprocess.Popen(
            f"redfold -test {fasta_dir}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        start_reading = False
        results = []
        for line in process.stdout:
            # print("line", line)
            if "Epoch Loss:" in line:
                start_reading = False
            if start_reading:
                results.append(line.strip())
            print(line, end="")  # Print live output
            if ">" in line:
                results = []
                start_reading = True
        print("results aaa", results)
        process.wait()

        # Ensure RedFold ran successfully
        if process.returncode != 0:
            raise RuntimeError(f"RedFold execution failed with exit code {process.returncode}")

        # Save the last two lines (sequence and structure) to result.db
        with open(output_file, "w") as f:
            f.write("\n".join(results) + "\n")

        logging.info(f"RedFold prediction completed. Results saved to {output_file}")

        return Response({
            'message': 'RedFold prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1
        }, status=status.HTTP_200_OK) 
    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict_redfold: {str(e)}")
        logging.error(f"Error in predict_redfold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in RedFold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_knotfold(fasta_path, output_dir):
    """Helper function to run KnotFold prediction and save results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the RedFold directory path
        knotfold_dir = os.path.join(os.path.dirname(__file__), "JiaLuModels/KnotFold")
        fasta_dir = os.path.dirname(fasta_path)

        # Construct the output file path
        output_file = os.path.join(output_dir, "result.db")

        # Set the environment variables
        env = os.environ.copy()
        os.chdir(knotfold_dir)
        env['PYTHONPATH'] = knotfold_dir + os.pathsep + env.get('PYTHONPATH', '')
        env['DATAPATH'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RNAstructure/data_tables")
        # Run RedFold with subprocess
        process = subprocess.Popen(
            f"python KnotFold_script.py -i {fasta_path} -o {output_dir} -d {env['DATAPATH']}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        start_reading = False
        results = []
        for line in process.stdout:
            # print("line", line)
            if "Epoch Loss:" in line:
                start_reading = False
            if start_reading:
                results.append(line.strip())
            print(line, end="")  # Print live output
            if ">" in line:
                results = []
                start_reading = True
        print("results", results)
        process.wait()

        # Ensure RedFold ran successfully
        if process.returncode != 0:
            raise RuntimeError(f"KnotFold execution failed with exit code {process.returncode}")

        # Save the last two lines (sequence and structure) to result.db
        with open(output_file, "w") as f:
            f.write("\n".join(results) + "\n")

        logging.info(f"KnotFold prediction completed. Results saved to {output_file}")

        return Response({
            'message': 'KnotFold prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1
        }, status=status.HTTP_200_OK) 

    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict_knotfold: {str(e)}")
        logging.error(f"Error in predict_knotfold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in KnotFold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def ct2dot(ct_path, output_file):
    """Helper function to run ct2dot prediction and save results"""
    try:

        env = os.environ.copy()
        env['DATAPATH'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RNAstructure/data_tables")
        env['PATH'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RNAstructure/exe") + os.pathsep + env.get('PATH', '')

        cmd = f"ct2dot {ct_path} 1 {output_file}"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            env=env
        )

        output = process.stdout.read()
        print(f"ct2dot output: {output}")
    
    except Exception as e:
        traceback.print_exc()
        print(f"Error in ct2dot: {str(e)}")
        logging.error(f"Error in ct2dot: {str(e)}", exc_info=True)

def predict_sincfold(fasta_path, output_dir):
    """Helper function to run RedFold prediction and save results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        ct_dir = os.path.join(os.path.dirname(output_dir), "ct")
        output_file = os.path.join(output_dir, "result.db")

        cmd = f"sincFold pred {fasta_path} -o {ct_dir}"
        print("cmd", cmd)
        # Run RedFold with subprocess
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output = process.stdout.read()
        print(f"sincFold output: {output}")
        for file in os.listdir(ct_dir):
            if file.endswith(".ct"):
                ct2dot(os.path.join(ct_dir, file), output_file)
        # Ensure RedFold ran successfully
        # if process.returncode != 0:
        #     raise RuntimeError(f"mxfold2 execution failed with exit code {process.returncode}")

        logging.info(f"sincFold prediction completed. Results saved to {output_file}")

        return Response({
            'message': 'sincFold prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1
        }, status=status.HTTP_200_OK) 

    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict sincFold: {str(e)}")
        logging.error(f"Error in predict sincFold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in sincFold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_mxfold2(fasta_path, output_dir):
    """Helper function to run RedFold prediction and save results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        output_file = os.path.join(output_dir, "result.db")

        cmd = f"mxfold2 predict {fasta_path} >{output_file}"
        print("cmd", cmd)
        # Run RedFold with subprocess
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        # Ensure RedFold ran successfully
        # if process.returncode != 0:
        #     raise RuntimeError(f"mxfold2 execution failed with exit code {process.returncode}")

        logging.info(f"mxfold2 prediction completed. Results saved to {output_file}")

        return Response({
            'message': 'mxfold2 prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1
        }, status=status.HTTP_200_OK) 

    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict_mxfold2: {str(e)}")
        logging.error(f"Error in predict_mxfold2: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in mxfold2 prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def predict_rnafold(fasta_path, output_dir):
    """Helper function to run RNAfold prediction and wait for output file"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the output file path
        output_file = os.path.join(output_dir, "result.db")
        
        # Run RNAfold with subprocess
        cmd = f"RNAfold < {fasta_path} > {output_file}"
        print("Running command:", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Wait for process to complete
        process.wait()
        
        # Read output
        output = process.stdout.read()
        print(f"RNAfold output: {output}")
        
        # Wait for output file to be generated with timeout
        max_wait_time = 60  # Maximum wait time in seconds
        wait_interval = 1   # Check interval in seconds
        waited_time = 0
        
        while not os.path.exists(output_file) and waited_time < max_wait_time:
            time.sleep(wait_interval)
            waited_time += wait_interval
            print(f"Waiting for output file... ({waited_time}s)")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not generated after {max_wait_time} seconds: {output_file}")
        
        # Ensure file is not empty
        if os.path.getsize(output_file) == 0:
            raise ValueError(f"Output file is empty: {output_file}")
            
        logging.info(f"RNAfold prediction completed. Results saved to {output_file}")
        
        return Response({
            'message': 'RNAfold prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1,
            'output_file': output_file
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict_rnafold: {str(e)}")
        logging.error(f"Error in predict_rnafold: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in RNAfold prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_RNAstructure(fasta_path, output_dir):
    """Helper function to run RNAstructure prediction and save results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Construct the output file path
        output_file = os.path.join(output_dir, "result.ct")
        shape_path = None
        for file in os.listdir(os.path.dirname(fasta_path)):
            if file.endswith(".shape"):
                shape_path = os.path.join(os.path.dirname(fasta_path), file)
                break

        python_path = os.path.join(os.path.dirname(__file__), 'JiaLuModels/RNAstructure.py')

        env = os.environ.copy()
        env["DATAPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RNAstructure/data_tables")

        cmd = f"python {python_path} {fasta_path} {output_file} --shape {shape_path} --si -0.4"
        print("cmd", cmd)   
        # Run RNAstructure with subprocess
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        print("process output", process.stdout.read())

        logging.info(f"RNAstructure prediction completed. Results saved to {output_file}")

        return Response({
            'message': 'RNAstructure prediction completed',
            'input_directory': fasta_path,
            'output_directory': output_dir,
            'files_processed': 1            
        }, status=status.HTTP_200_OK) 

    except Exception as e:
        traceback.print_exc()
        print(f"Error in predict_RNAstructure: {str(e)}")
        logging.error(f"Error in predict_RNAstructure: {str(e)}", exc_info=True)
        return Response({   
            'error': f'Error in RNAstructure prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def predict_seismic(fasta_path, output_dir, paired_end_reads_dir):
    """Helper function to run SEISMIC prediction with paired-end FASTQ files"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        python_path = os.path.join(os.path.dirname(__file__), 'JiaLuModels/seismic.py')
        
        # Construct the output file path
        temp_output_dir = os.path.join(os.path.dirname(output_dir), "temp_output")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Build the command to run SEISMIC
        cmd = f"python {python_path} --fastq_dir {paired_end_reads_dir} --fasta_file {fasta_path}  --min_clusters 1 --max_clusters 3 --output ./"
        print(f"Running SEISMIC command: {cmd}")

        env = os.environ.copy()
        env["DATAPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RNAstructure/data_tables")
        
        # Change to the SEISMIC directory before running the script
        current_dir = os.getcwd()
        os.chdir(temp_output_dir)
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Wait for process to complete
            process.wait()
            
            # Read output
            output = process.stdout.read()
            print(f"SEISMIC output: {output}")
            for file in os.listdir(os.path.join(temp_output_dir, "out")):
                for result_file in os.listdir(os.path.join(temp_output_dir, "out", file, "fold/rre/full")):
                    if result_file.endswith(".db"):
                        # copy the file to the output directory
                        shutil.copy(os.path.join(temp_output_dir, "out", file, "fold/rre/full", result_file), os.path.join(output_dir, result_file))
                
            logger.info(f"SEISMIC prediction completed. Results saved to {output_dir}")
            
            return Response({
                'message': 'SEISMIC prediction completed',
                'input_directory': fasta_path,
                'output_directory': output_dir,
                'files_processed': 1,
                'output_file': output_dir
            }, status=status.HTTP_200_OK)
            
        finally:
            # Always restore the original working directory
            os.chdir(current_dir)
        
    except Exception as e:
        logger.error(f"Error in predict_seismic: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error in SEISMIC prediction: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def run_miRNA(fasta_path, output_dir):
    """Helper function to run miRNA prediction"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        model_dir = os.path.join(os.path.dirname(__file__), "AnalysisTools/mirdb-custom-target-search")
        species = "Human"
        conda_env = "shapeRNA"
        conda_exe = "/pubapps/qsong1/miniconda3/bin/conda"
        os.chdir(model_dir)

        # remove the output file if it exists
        if os.path.exists(os.path.join(model_dir, "test.csv")):
            os.remove(os.path.join(model_dir, "test.csv"))
        if os.path.exists(os.path.join(model_dir, "test.db")):
            os.remove(os.path.join(model_dir, "test.db"))

        cmd = (
            f"{conda_exe} run -n shapeRNA python miRNA.py --mirdb_script mirdb_custom_target_search.py --fasta {fasta_path} --species {species} --conda_env {conda_env}"
        )
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output = process.stdout.read()
        print(f"miRNA output: {output}")    

        # move the output file (test.csv) to the output directory
        shutil.move(os.path.join(model_dir, "test.csv"), os.path.join(output_dir, "miRNA_result.csv"))
        current_dir = os.getcwd()
        os.chdir(current_dir)

        logging.info(f"miRNA prediction completed. Results saved to {output_dir}")

        return True
    except Exception as e:
        logger.error(f"Error in run_miRNA: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False

def run_deepsramp(fasta_path, output_dir):
    """Helper function to run DeepSRAMP prediction"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        os.chdir(os.path.join(os.path.dirname(__file__), "AnalysisTools/sramp_simple"))

        output_file = os.path.join(output_dir, "m6a_result.csv")

        cmd = (f"perl runsramp.pl  {fasta_path} {output_file} full")
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print(f"DeepSRAMP output: {output}")

        logging.info(f"DeepSRAMP prediction completed. Results saved to {output_file}")

        return True
    except Exception as e:
        logger.error(f"Error in run_deepsramp: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False

def run_rbpmap(fasta_path, output_dir):
    """Helper function to run RBPmap prediction"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)  
        # base conda env
        conda_exe = "/pubapps/qsong1/miniconda3/bin/conda"
        conda_env = "shapeRNA"
        os.chdir(os.path.join(os.path.dirname(__file__), "AnalysisTools/RBPmap"))

        cmd = f"{conda_exe} run --no-capture-output -n {conda_env} python rbpmap.py --rbpmap_script ~/rbpmap/RBPmap.pl --input {fasta_path} --db mm10"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print(f"RBPmap output: {output}")

        for output_line in output.split("\n"):
            if "The results can be found under:" in output_line:
                predict_file = os.path.join(output_line.split("The results can be found under:")[1].strip(), "All_Predictions.txt")

                # copy the result_dir to the output directory
                shutil.copy(predict_file, output_dir)
                break
        else:
            raise Exception("No results found in RBPmap output")

        logging.info(f"RBPmap prediction completed. Results saved to {output_dir}")

        return True
    except Exception as e:
        logger.error(f"Error in run_rbpmap: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False

def generate_visualization_helper(output_dir):
    """Helper function to generate visualization"""
    try:
        input_file = os.path.join(output_dir, "result.db")
        output_file = os.path.join(output_dir, "rna_structure.png")
        # Run VARNA to generate visualization
        cmd = f"java -cp {os.path.join(os.path.dirname(__file__), 'AnalysisTools/visualization/VARNAv3-93.jar')} fr.orsay.lri.varna.applications.VARNAcmd -i {input_file} -o {output_file}"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print(f"VARNA output: {output}")

        return True
    except Exception as e:
        logger.error(f"Error in generate_visualization_helper: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False

@api_view(['POST'])
def upload_fasta(request):
    try:
        if 'files' not in request.FILES:
            print("No files in request.FILES")
            logger.error("No files in request.FILES")
            return Response({'error': 'No files uploaded'}, 
                        status=status.HTTP_400_BAD_REQUEST)
        
        selected_model = request.POST.get('model')
        if not selected_model:
            print("No model selected")
            logger.error("No model selected")
            return Response({'error': 'No model selected'}, 
                        status=status.HTTP_400_BAD_REQUEST)
        
        # Check if this is a visualization-only request
        is_visualization_only = selected_model.lower() == 'visualization_only'
        
        # Check if SHAPE data is being used
        use_shape = request.POST.get('use_shape', 'false').lower() == 'true'
        
        # Check if seismic model is selected
        is_seismic_model = selected_model.lower() == 'seismic'
        
        # If SHAPE is required and not seismic, check for SHAPE file
        if use_shape and not is_seismic_model and not is_visualization_only and 'shape_files' not in request.FILES:
            print("SHAPE file required but not provided")
            logger.error("SHAPE file required but not provided")
            return Response({'error': 'SHAPE file is required'}, 
                        status=status.HTTP_400_BAD_REQUEST)
        
        # If seismic model is selected, check for FASTQ files
        if use_shape and is_seismic_model:
            if 'fastq_r1_files' not in request.FILES:
                print("FASTQ R1 file required but not provided")
                logger.error("FASTQ R1 file required but not provided")
                return Response({'error': 'FASTQ R1 file is required'}, 
                            status=status.HTTP_400_BAD_REQUEST)
            
            if 'fastq_r2_files' not in request.FILES:
                print("FASTQ R2 file required but not provided")
                logger.error("FASTQ R2 file required but not provided")
                return Response({'error': 'FASTQ R2 file is required'}, 
                            status=status.HTTP_400_BAD_REQUEST)
        
        print("use_shape", use_shape)
        print("is_seismic_model", is_seismic_model)
        print("is_visualization_only", is_visualization_only)
        
        # Create timestamp for unique folder names
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = os.path.join(os.path.dirname(__file__), "UCE_DB")
        timestamp_dir = os.path.join(base_path, current_time)
        input_dir = os.path.join(timestamp_dir, "input")
        output_dir = os.path.join(timestamp_dir, "output")
        fasta_dir = os.path.join(timestamp_dir, "fasta")
        data_dir = os.path.join(input_dir, "data")
        save_model_dir = os.path.join(timestamp_dir, "model")
        paired_end_reads_dir = os.path.join(timestamp_dir, "paired_end_reads")
        
        # Add debug logging
        logger.debug(f"Base path: {base_path}")
        logger.debug(f"Input dir: {input_dir}")
        logger.debug(f"Output dir: {output_dir}")
        logger.debug(f"Using SHAPE data: {use_shape}")
        logger.debug(f"Is seismic model: {is_seismic_model}")
        logger.debug(f"Is visualization only: {is_visualization_only}")
        
        # Clean up old folders before creating new ones
        cleanup_old_folders(base_path)
        
        # Create all necessary directories
        os.makedirs(timestamp_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fasta_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(save_model_dir, exist_ok=True)
        
        # Create paired_end_reads directory if using seismic model
        if is_seismic_model:
            os.makedirs(paired_end_reads_dir, exist_ok=True)

        # Handle the uploaded file
        uploaded_file = request.FILES['files']
        logger.debug(f"Received file: {uploaded_file.name}, size: {uploaded_file.size}")
        
        # For visualization-only mode, save the file directly to the output directory
        if is_visualization_only:
            result_db_path = os.path.join(output_dir, 'result.db')
            with default_storage.open(result_db_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
                    
            return Response({
                'message': 'File uploaded successfully for visualization',
                'input_directory': input_dir,
                'output_directory': output_dir,
                'files_processed': 1
            }, status=status.HTTP_200_OK)
        
        # For normal mode, save the FASTA file
        fasta_path = os.path.join(fasta_dir, 'sequence.fasta')
        with default_storage.open(fasta_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Handle SHAPE file if provided (for non-seismic models)
        shape_path = None
        if use_shape and not is_seismic_model and 'shape_files' in request.FILES:
            shape_file = request.FILES['shape_files']
            logger.debug(f"Received SHAPE file: {shape_file.name}, size: {shape_file.size}")
            
            # Save the SHAPE file
            shape_path = os.path.join(fasta_dir, 'sequence.shape')
            with default_storage.open(shape_path, 'wb+') as destination:
                for chunk in shape_file.chunks():
                    destination.write(chunk)
        
        # Handle FASTQ files if using seismic model
        fastq_r1_path = None
        fastq_r2_path = None
        if use_shape and is_seismic_model:
            # Handle R1 file
            fastq_r1_file = request.FILES['fastq_r1_files']
            logger.debug(f"Received FASTQ R1 file: {fastq_r1_file.name}, size: {fastq_r1_file.size}")
            
            # Save the FASTQ R1 file
            fastq_r1_path = os.path.join(paired_end_reads_dir, 'reads_R1.fq.gz')
            with default_storage.open(fastq_r1_path, 'wb+') as destination:
                for chunk in fastq_r1_file.chunks():
                    destination.write(chunk)
            
            # Handle R2 file
            fastq_r2_file = request.FILES['fastq_r2_files']
            logger.debug(f"Received FASTQ R2 file: {fastq_r2_file.name}, size: {fastq_r2_file.size}")
            
            # Save the FASTQ R2 file
            fastq_r2_path = os.path.join(paired_end_reads_dir, 'reads_R2.fq.gz')
            with default_storage.open(fastq_r2_path, 'wb+') as destination:
                for chunk in fastq_r2_file.chunks():
                    destination.write(chunk)
        if not run_miRNA(fasta_path, output_dir):
            return Response({
                'error': 'Error in miRNA prediction'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not run_deepsramp(fasta_path, output_dir):
            return Response({
                'error': 'Error in DeepSRAMP prediction'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not run_rbpmap(fasta_path, output_dir):
            return Response({
                'error': 'Error in RBPmap prediction'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Call the appropriate prediction function based on the selected model
        if selected_model.lower() == 'seismic':
            return predict_seismic(fasta_path, output_dir, paired_end_reads_dir)
        elif selected_model.lower() == 'linearsampling':
            return predict_linearsampling(fasta_path, output_dir, use_shape=use_shape, shape_path=shape_path)
        elif selected_model.lower() == 'ufold':
            return predict_ufold(fasta_path, output_dir)
        elif selected_model.lower() == 'cnnfold':
            return predict_cnnfold(fasta_path, output_dir)
        elif selected_model.lower() == 'knotfold':
            return predict_knotfold(fasta_path, output_dir)
        elif selected_model.lower() == 'redfold':
            return predict_redfold(fasta_path, output_dir)
        elif selected_model.lower() == 'sincfold':
            return predict_sincfold(fasta_path, output_dir)
        elif selected_model.lower() == 'mxfold2':
            return predict_mxfold2(fasta_path, output_dir)
        elif selected_model.lower() == 'rnafold':
            return predict_rnafold(fasta_path, output_dir)
        elif selected_model.lower() == 'rnastructure':
            return predict_RNAstructure(fasta_path, output_dir)
        else:
            return Response({
                'error': f'Unsupported model: {selected_model}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        print(f"Error in upload_fasta: {str(e)}")
        traceback.print_exc()
        return Response({
            'error': f'Error processing FASTA file: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])     
def generate_visualization(request):
    """API endpoint to generate visualization"""
    try:
        output_dir = request.data.get('output_dir')
        if not output_dir:
            return Response({
                'error': 'No output directory provided'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Call the helper function to generate the visualization
        if not generate_visualization_helper(output_dir):
            return Response({
                'error': 'Error in visualization generation'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Return the path to the generated image
        image_path = os.path.join(output_dir, "rna_structure.png")
        image_url = f"/backend/api/download-result/?output_dir={os.path.dirname(image_path)}&file=rna_structure.png"

        helix_image_path = generate_helix_visualization(output_dir)
        if helix_image_path:
            helix_image_file = os.path.basename(helix_image_path)
            helix_image_url = f"/backend/api/download-result/?output_dir={os.path.dirname(helix_image_path)}&file={helix_image_file}"
        else:
            print("Some error in helix visualization generation")
            helix_image_url = None
        
        return Response({
            'message': 'Visualization generated successfully',
            'image_url': image_url,
            'helix_image_url': helix_image_url
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in generate_visualization: {str(e)}", exc_info=True)
        traceback.print_exc()
        return Response({
            'error': f'Error in visualization generation: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def get_first_sequence_name(fasta_path):
    with open(fasta_path, 'r') as file:
        for line in file:
            if line.startswith(">"):
                return line[1:].strip().split()[0]  # Extract sequence name
    return None  # Return None if no sequence is found

def pdf_to_png(pdf_path, ouput_image_path, dpi=300):
    try:
        # Convert PDF to a list of images
        images = convert_from_path(pdf_path, dpi=dpi)

        # Save each page as PNG
        for i, img in enumerate(images):
            img.save(ouput_image_path, "PNG")
            print(f"Saved: {ouput_image_path}")
            return True
    except Exception as e:
        logger.error(f"Error in pdf_to_png: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False

def generate_helix_visualization(output_dir):
    """Helper function to generate helix visualization"""
    try:

        rbp_file = os.path.join(output_dir, "All_Predictions.txt")
        mirna_file = os.path.join(output_dir, "miRNA_result.csv")
        m6a_file = os.path.join(output_dir, "m6a_result.csv")

        if not os.path.exists(rbp_file) or not os.path.exists(mirna_file) or not os.path.exists(m6a_file):
            return False

        base_dir = os.path.dirname(output_dir)
        fasta_file = os.path.join(base_dir, "fasta", "sequence.fasta")
        first_sequence_name = get_first_sequence_name(fasta_file)
        data_table_dir = os.path.join(base_dir, "data_table")

        if not os.path.exists(data_table_dir):
            os.makedirs(data_table_dir)

        #Step1:进行数据表的生成
        python_dir = os.path.join(os.path.dirname(__file__), "AnalysisTools/visualization")
        os.chdir(python_dir)
        cmd = f"python data_table.py --rbp {rbp_file} --mirna {mirna_file} --output_dir {data_table_dir}"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print("Step1: data table output", output)

        #Step2:进行m6a相关的预测，并将文件拆分并存入folder中
        sramp_path = os.path.join(os.path.dirname(__file__), "AnalysisTools/sramp_simple")
        m6a_python_path = os.path.join(os.path.dirname(__file__), "AnalysisTools/visualization/m6a-new.py")
        m6a_output_dir = os.path.join(base_dir, "m6a_output")
        if not os.path.exists(m6a_output_dir):
            os.makedirs(m6a_output_dir)
        os.chdir(sramp_path)
        cmd = f"python {m6a_python_path} --fasta {fasta_file} --runsramp runsramp.pl --output_dir {m6a_output_dir}"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print("Step2: m6a output", output) 

        #Step3:按照前面处理出的miRNA，rbp和m6a的信息做成标注文件annotation.txt
        python_dir = os.path.join(os.path.dirname(__file__), "AnalysisTools/visualization")
        os.chdir(python_dir)
        annotation_file = os.path.join(output_dir, "annotations.txt")
        cmd = f"python annotation.py --m6a {m6a_file} --rbp_dir {m6a_output_dir} --mirna {m6a_output_dir} --output {annotation_file}"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print("Step3: annotation output", output)

        # copy the result.db to the m6a_output folder and rename it as the first sequence name
        result_db_path = os.path.join(output_dir, "result.db")
        shutil.copy(result_db_path, os.path.join(m6a_output_dir, f"{first_sequence_name}.db"))

        #Step4:进行helix的预测
        cmd = f"Rscript helix.R {annotation_file} {m6a_output_dir} {output_dir}"
        print("cmd", cmd)
        
        process = subprocess.Popen(
            cmd,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = process.stdout.read()
        print("Step4: helix output", output)
        os.chdir(output_dir)
        # convert the helix.pdf to png
        pdf_file = os.path.join(output_dir, f"{first_sequence_name}_double_helix.pdf")
        png_file = os.path.join(output_dir, f"{first_sequence_name}_double_helix.png")
        if not pdf_to_png(pdf_file, png_file):
            print("Some error in helix visualization generation")
            return False
        return png_file
    except Exception as e:
        logger.error(f"Error in generate_helix_visualization: {str(e)}", exc_info=True)
        traceback.print_exc()
        return False


@api_view(['POST'])
def clear_cache(request):
    """
    Clear temporary files and directories from a specific run
    """
    try:
        input_directory = request.data.get('input_directory')
        output_directory = request.data.get('output_directory')
        shutil.rmtree(os.path.dirname(output_directory))
        return Response({
            'message': 'Cache cleared successfully'
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in clear_cache: {str(e)}", exc_info=True)
        return Response({
            'error': f'Error clearing cache: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
