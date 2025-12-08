import astropy.io.fits as fits
import os
import traceback
from astropy.time import Time

def get_sorted_files(folder):
    all_items = os.listdir(folder)
    files = [
        os.path.join(folder, item) 
        for item in all_items 
        if os.path.isfile(os.path.join(folder, item))
    ]
    files.sort(key=lambda x: os.path.basename(x))
    return files

if __name__ == '__main__':
    # srcSPL2 = 'E:/Research/Data/HINODE/SP/L2/2014/' # for test
    srcSPL2 = 'E:/Research/Data/HINODE/SP/L2/2019/'
    # srcSPL2_rd = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/SPL2_reduced/2014' # for test
    srcSPL2_rd = 'E:/Research/Work/Magnetogram_SuperResol_by_NN/SPL2_reduced/2019/'
    # year_record = '2014' # for test
    year_record = '2019'
    # Failure log path (saved in the reduced data directory)
    failed_log_path = os.path.join(srcSPL2_rd, f'processing_failed_{year_record}.txt')

    if not os.path.exists(srcSPL2_rd):
        os.mkdir(srcSPL2_rd)
    
    # Initialize failure log (append mode, no overwriting)
    with open(failed_log_path, 'a', encoding='utf-8') as failed_log:
        # Write log header only if file is empty
        if os.path.getsize(failed_log_path) == 0:
            failed_log.write(f"===== Processing Failure Log ({year_record}) =====\n")
            failed_log.write("Log Time Format: UTC\n")
            failed_log.write("Format: Filename | Failure Time | Error Message\n")
            failed_log.write("-" * 80 + "\n")
        
        # Sorting file folder
        filesSPL2 = get_sorted_files(srcSPL2)
        
        # Iterating throughout all files
        for i_file in range(len(filesSPL2)):
            
            print(f'Begin processing No.{i_file+1} files')
            
            fnSPL2 = filesSPL2[i_file]
            file_basename = os.path.basename(fnSPL2)
            current_utc = Time.now().isot  # Record current UTC time for log
            SPL2 = None  # Initialize FITS object
            
            try:
                RecTime = fnSPL2[-20:-5]
                
                SPL2 = fits.open(os.path.join(srcSPL2, fnSPL2))
           
                def reduce_SPL2(fn, SPL2):
                    # Saving the data containing grid and slit position
                    grid_hdu= [SPL2[idx] for idx in [38, 39, 41]]
                    prim_hdu = fits.PrimaryHDU()
                    hdu_list = [prim_hdu] + grid_hdu
                    fits.HDUList(hdu_list).writeto(os.path.join(srcSPL2_rd, fn), overwrite=True)
                
                reduce_SPL2(RecTime + '.fits', SPL2)
                print(f'Reduce SP-L2 files to {RecTime}.fits')
                SPL2.close()  # Close FITS file after successful processing
            
            except Exception as e:
                # Capture detailed error information (truncate long traceback for readability)
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[:500]}..."
                print(f"Error processing {file_basename}: {str(e)}, skipping to next file")
                
                # Write failure record to log (replace '|' to avoid format confusion)
                failed_log.write(f"{file_basename} | {current_utc} | {error_msg.replace('|', '-')}\n")
                failed_log.flush()  # Flush immediately to avoid data loss
                
                # Ensure FITS file is closed even if error occurs
                try:
                    if SPL2 is not None:
                        SPL2.close()
                except:
                    pass

    print(f"\nProcessing Completed\nFailure log saved to: {failed_log_path}")