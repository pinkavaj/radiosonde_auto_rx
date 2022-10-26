#!/usr/bin/env python
#
#   radiosonde_auto_rx - SDR Abstraction
#
#   Copyright (C) 2022  Mark Jessop <vk5qi@rfhead.net>
#   Released under GNU GPL v3 or later
#
import logging
import os.path
import platform
import subprocess
import numpy as np
from datetime import datetime
import numpy.fft as fft

from .utils import rtlsdr_test, reset_rtlsdr_by_serial, reset_all_rtlsdrs


def test_sdr(
    sdr_type: str,
    rtl_device_idx = "0",
    sdr_hostname = "",
    sdr_port = 5555,
    ss_iq_path = "./ss_iq",
    ss_power_path = "./ss_power"
):
    """
    Test the prescence / functionality of a SDR.

    sdr_type (str): 'RTLSDR', 'SpyServer' or 'KA9Q'

    Arguments for RTLSDRs:
    rtl_device_id (str) - Device ID for a RTLSDR

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server

    Arguments for SpyServer Client:
    ss_iq_path (str): Path to spyserver IQ client utility.
    ss_power_path (str): Path to spyserver power utility.
    """

    if sdr_type == "RTLSDR":
        # Use the existing rtlsdr_test code, which tries to read some samples
        # from the specified RTLSDR
        _ok = rtlsdr_test(rtl_device_idx)
        if not _ok:
            logging.error(f"RTLSDR #{rtl_device_idx} non-functional.")

        return _ok


    elif sdr_type == "KA9Q":
        _ok = True
        # TODO: add proper check

        if not _ok:
            logging.error(f"KA9Q Server {sdr_hostname}:{sdr_port} non-functional.")

        return _ok

    elif sdr_type == "SpyServer":
        # Test connectivity to a SpyServer by trying to grab some samples.

        if not os.path.isfile(ss_iq_path):
            logging.critical("Could not find ss_iq binary! This may need to be compiled.")
            return False

        _cmd = (
            f"timeout 10 "  # Add a timeout, because connections to non-existing IPs seem to block.
            f"{ss_iq_path} "
            f"-f 401500000 "
            f"-s 48000 "
            f"-r {sdr_hostname} -q {sdr_port} -n 48000 - > /dev/null 2> /dev/null"
        )

        logging.debug(f"SpyServer - Testing using command: {_cmd}")

        try:
            _output = subprocess.check_output(
                _cmd, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            # Something went wrong...
            logging.critical(
                f"SpyServer ({sdr_hostname}:{sdr_port}) - ss_iq call failed with return code {e.returncode}."
            )
            return False

        return True

    else:
        logging.error(f"Test SDR: Unknown SDR Type {sdr_type}")
        return False



def reset_sdr(
    sdr_type: str,
    rtl_device_idx = "0",
    sdr_hostname = "",
    sdr_port = 5555
    ):
    """
    Attempt to reset a SDR. Only used for RTLSDRs.

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'

    Arguments for RTLSDRs:
    rtl_device_id (str) - Device ID for a RTLSDR

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server
    """

    if sdr_type == "RTLSDR":
        # Attempt to reset the RTLSDR.
        if rtl_device_idx == "0":
            # If the device ID is 0, we assume we only have a single RTLSDR on this system.
            reset_all_rtlsdrs()
        else:
            # Otherwise, we reset the specific RTLSDR
            reset_rtlsdr_by_serial(rtl_device_idx)

    else:
        logging.debug(f"No reset ability for SDR type {sdr_type}")


def get_sdr_name(
    sdr_type: str,
    rtl_device_idx = "0",
    sdr_hostname = "",
    sdr_port = 5555
    ):
    """
    Get a human-readable name of the currenrt SDR

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'

    Arguments for RTLSDRs:
    rtl_device_id (str) - Device ID for a RTLSDR

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server
    """

    if sdr_type == "RTLSDR":
        return f"RTLSDR {rtl_device_idx}"

    elif sdr_type == "KA9Q":
        return f"KA9Q {sdr_hostname}:{sdr_port}"

    elif sdr_type == "SpyServer":
        return f"SpyServer {sdr_hostname}:{sdr_port}"

    else:
        return f"UNKNOWN {sdr_type}"


def shutdown_sdr(
    sdr_type: str,
    sdr_id: str
    ):
    """
    Function to trigger shutdown/cleanup of some SDR types.
    Currently only required for the KA9Q server.

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'
    sdr_id (str): The global ID of the SDR to be shut down.
    """

    if sdr_type == "KA9Q":
        # TODO - KA9Q Server channel cleanup.
        logging.debug(f"TODO - Cleanup for SDR type {sdr_type}")
    else:
        logging.debug(f"No shutdown action required for SDR type {sdr_type}")

    return



def get_sdr_iq_cmd(
    sdr_type: str,
    frequency: int,
    sample_rate: int,
    rtl_device_idx = "0",
    rtl_fm_path = "rtl_fm",
    fast_filter: bool = False,
    ppm = 0,
    gain = None,
    bias = False,
    sdr_hostname = "",
    sdr_port = 5555,
    ss_iq_path = "./ss_iq"
):
    """
    Get a command-line argument to get IQ (signed 16-bit) from a SDR
    for a given frequency and bandwidth.

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'
    frequency (int): Centre frequency in Hz
    sample_rate (int): Sample rate in Hz

    Arguments for RTLSDRs:
    rtl_device_idx (str) - Device ID for a RTLSDR
    rtl_fm_path (str) - Path to rtl_fm. Defaults to just "rtl_fm"
    ppm (int): SDR Frequency accuracy correction, in ppm.
    gain (int): SDR Gain setting, in dB. A gain setting of -1 enables the RTLSDR AGC.
    bias (bool): If True, enable the bias tee on the SDR.
    fast_filter (bool): If true, drop the -F9 higher quality filter for rtl_fm

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server

    Arguments for SpyServer Client:
    ss_iq_path (str): Path to spyserver IQ client utility.

    """

    if sdr_type == "RTLSDR":
        _gain = ""
        if gain:
            if gain >= 0:
                _gain = f"-g {gain:.1f} "

        _cmd = (
            f"{rtl_fm_path} -M raw "
            f"{'' if fast_filter else '-F9 '}"
            f"{'-T ' if bias else ''}"
            f"-p {int(ppm)} "
            f"-d {str(rtl_device_idx)} "
            f"{_gain}"
            f"-s {int(sample_rate)} "
            f"-f {int(frequency)} "
            f"- 2>/dev/null | "
        )

        return _cmd

    if sdr_type == "SpyServer":
        _cmd = (
            f"{ss_iq_path} "
            f"-f {frequency} "
            f"-s {int(sample_rate)} "
            f"-r {sdr_hostname} -q {sdr_port} - 2>/dev/null|"
        )

        return _cmd

    if sdr_type == "KA9Q":
        raise NotImplementedError(f"Not implemented for sdr_type={sdr_type}")

    else:
        raise NotImplementedError(f"Not implemented for sdr_type={sdr_type}")



def get_sdr_fm_cmd(
    sdr_type: str,
    frequency: int,
    filter_bandwidth: int,
    sample_rate: int,
    highpass = None,
    lowpass = None,
    rtl_device_idx = "0",
    rtl_fm_path = "rtl_fm",
    ppm = 0,
    gain = -1,
    bias = False,
    sdr_hostname = "",
    sdr_port = 1234,
):
    """
    Get a command-line argument to get FM demodulated audio (signed 16-bit) from a SDR
    for a given frequency and bandwidth.

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'
    frequency (int): Centre frequency in Hz
    filter_bandwidth (int): FM Demodulator filter bandwidth in Hz
    sample_rate (int): Output sample rate in Hz

    Optional arguments
    highpass (int): If provided, add a high-pass filter after the FM demodulator.
    lowpass (int): If provided, add a low-pass filter after the FM demodulator.

    Arguments for RTLSDRs:
    rtl_device_idx (str) - Device ID for a RTLSDR
    rtl_fm_path (str) - Path to rtl_fm. Defaults to just "rtl_fm"
    ppm (int): SDR Frequency accuracy correction, in ppm.
    gain (int): SDR Gain setting, in dB. A gain setting of -1 enables the RTLSDR AGC.
    bias (bool): If True, enable the bias tee on the SDR.

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server

    """


    if sdr_type == "RTLSDR":
        _gain = ""
        if gain:
            if gain >= 0:
                _gain = f"-g {gain:.1f} "

        _cmd = (
            f"{rtl_fm_path} -M fm -F9 "
            f"{'-T ' if bias else ''}"
            f"-p {int(ppm)} "
            f"-d {str(rtl_device_idx)} "
            f"{_gain}"
            f"-s {int(filter_bandwidth)} "
            f"-f {int(frequency)} "
            f"2>/dev/null | "
        )

        # Add in resampling / convertion to wav using sox.
        _cmd += f"sox -t raw -r {int(filter_bandwidth)} -e s -b 16 -c 1 - -r {int(sample_rate)} -b 16 -t wav - "

        if highpass:
            _cmd += f"highpass {int(highpass)} "

        if lowpass:
            _cmd += f"lowpass {int(lowpass)} "

        _cmd += "2> /dev/null | "

        return _cmd

    else:
        raise NotImplementedError(f"Not implemented for sdr_type={sdr_type}")


def read_rtl_power_log(log_filename, sdr_name):
    """
    Read in a rtl_power compatible log output file.

    Arguments:
    log_filename (str): Filename to read
    sdr_name (str): SDR name used for logging errors.
    """

    # OK, now try to read in the saved data.
    # Output buffers.
    freq = np.array([])
    power = np.array([])

    freq_step = 0

    # Open file.
    f = open(log_filename, "r")

    # rtl_power log files are csv's, with the first 6 fields in each line describing the time and frequency scan parameters
    # for the remaining fields, which contain the power samples.

    for line in f:
        # Split line into fields.
        fields = line.split(",", 6)

        if len(fields) < 6:
            logging.error(
                f"Scanner ({sdr_name}) - Invalid number of samples in input file - corrupt?"
            )
            raise Exception(
                f"Scanner ({sdr_name}) - Invalid number of samples in input file - corrupt?"
            )

        start_date = fields[0]
        start_time = fields[1]
        start_freq = float(fields[2])
        stop_freq = float(fields[3])
        freq_step = float(fields[4])
        n_samples = int(fields[5])
        # freq_range = np.arange(start_freq,stop_freq,freq_step)
        samples = np.fromstring(fields[6], sep=",")
        freq_range = np.linspace(start_freq, stop_freq, len(samples))

        # Add frequency range and samples to output buffers.
        freq = np.append(freq, freq_range)
        power = np.append(power, samples)

    f.close()

    # Sanitize power values, to remove the nan's that rtl_power puts in there occasionally.
    power = np.nan_to_num(power)

    return (freq, power, freq_step)


def ka9q_get_channel_info():
    output = subprocess.check_output(
        '/home/pinky/work/air/ka9q-radio/metadump -1 sdr1-ctl.local',
        shell=True,
    ).decode()
    # Thu 20 Oct 2022 22:49:04.080492 CEST pinky-a:59660 STAT: (1) cmd tag 0 (2) commands 0 (3) Fri 21 Oct 2022 00:49:04.080382 CEST (16) out data src pinky-a:40023 (17) out data dst sdr1.local:5004 (18) out SSRC 1 666 298 019 (19) out TTL 1 (10) in samprate 2 400 000 Hz (22) out data pkts 30 497 405 (21) out metadata pkts 75 876 (69) output level -30,1 dB (24) calibration 0 (28) DC I offset 0 (68) gain 0,0 dB (33) RF 401 000 001,526 Hz (78) lock 1 (48) demod 0 (linear) (20) out samprate 2 400 000 Hz (49) out channels 2 (40) filt high 1,128e+06 Hz (39) filt low -1,128e+06 Hz (82) output bits/sample 8 (32) direct conv 1

    freq = output.split(' (33) RF ')[1].split(' Hz ', maxsplit=1)[0].replace('\u202f', '')
    sample_rate = output.split(' (20) out samprate ')[1].split(' Hz ', maxsplit=1)[0].replace('\u202f', '')
    return {
        'freq': int(float(freq.replace(',', '.'))),
        'sample_rate': int(sample_rate),
    }


def get_power_spectrum(
    sdr_type: str,
    frequency_start: int = 400050000,
    frequency_stop: int = 403000000,
    step: int = 800,
    integration_time: int = 20,
    rtl_device_idx = "0",
    rtl_power_path = "rtl_power",
    ka9q_iqrecord_path = "/home/pinky/work/air/ka9q-radio/iqrecord",
    ppm = 0,
    gain = None,
    bias = False,
    sdr_hostname = "",
    sdr_port = 5555,
    ss_power_path = "./ss_power"
):
    """
    Get power spectral density data from a SDR.

    Arguments:

    sdr_type (str): 'RTLSDR', 'Spyserver' or 'KA9Q'

    frequency_start (int): Start frequency for the PSD, Hz
    frequency_stop (int): Stop frequency for the PSD, Hz
    step (int): Requested frequency step for the PSD, Hz. May not always be honoured.
    integration_time (int): Integration time in seconds.

    Arguments for RTLSDRs:
    rtl_device_idx (str): Device ID for a RTLSDR
    rtl_power_path (str): Path to rtl_power. Defaults to just "rtl_power"
    ppm (int): SDR Frequency accuracy correction, in ppm.
    gain (int): SDR Gain setting, in dB. A gain setting of -1 enables the RTLSDR AGC.
    bias (bool): If True, enable the bias tee on the SDR.

    Arguments for KA9Q SDR Server / SpyServer:
    sdr_hostname (str): Hostname of KA9Q Server
    sdr_port (int): Port number of KA9Q Server

    Arguments for SpyServer Client:
    ss_power_path (str): Path to spyserver power utility.
    ss_iq_path (str): Path to spyserver IQ client utility.


    Returns:
    (freq, power, step) Tuple

    freq (np.array): Array of frequencies, in Hz
    power (np.array): Array of uncalibrated power estimates, in Hz
    step (float): Frequency step of the output data, in Hz.

    Returns (None, None, None) if an error occurs.

    """

    # No support for getting spectrum data on any other SDR source right now.
    # Override sdr selection.


    if sdr_type == "RTLSDR":
        # Use rtl_power to obtain power spectral density data

        # Create filename to output to.
        _log_filename = f"log_power_{rtl_device_idx}.csv"

        # If the output log file exists, remove it.
        if os.path.exists(_log_filename):
            os.remove(_log_filename)


        # Add -k 30 option, to SIGKILL rtl_power 30 seconds after the regular timeout expires.
        # Note that this only works with the GNU Coreutils version of Timeout, not the IBM version,
        # which is provided with OSX (Darwin).
        _platform = platform.system()
        if "Darwin" in _platform:
            _timeout_kill = ""
        else:
            _timeout_kill = "-k 30 "

        _timeout_cmd = f"timeout {_timeout_kill}{integration_time+10}"

        _gain = ""
        if gain:
            if gain >= 0:
                _gain = f"-g {gain:.1f} "

        _rtl_power_cmd = (
            f"{_timeout_cmd} {rtl_power_path} "
            f"{'-T ' if bias else ''}"
            f"-p {int(ppm)} "
            #f"-d {str(rtl_device_idx)} "
            f"{_gain}"
            f"-f {frequency_start}:{frequency_stop}:{step} "
            f"-i {integration_time} -1 -c 25% "
            f"{_log_filename}"
        )

        _sdr_name = get_sdr_name(
            sdr_type=sdr_type,
            rtl_device_idx=rtl_device_idx,
            sdr_hostname=sdr_hostname,
            sdr_port=sdr_port
            )

        logging.info(f"Scanner ({_sdr_name}) - Running frequency scan.")
        logging.debug(
            f"Scanner ({_sdr_name}) - Running command: {_rtl_power_cmd}"
        )

        try:
            _output = subprocess.check_output(
                _rtl_power_cmd, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            # Something went wrong...
            logging.critical(
                f"Scanner ({_sdr_name}) - rtl_power call failed with return code {e.returncode}."
            )
            # Look at the error output in a bit more details.
            _output = e.output.decode("ascii")
            if "No supported devices found" in _output:
                logging.critical(
                    f"Scanner ({_sdr_name}) - rtl_power could not find device with ID {rtl_device_idx}, is your configuration correct?"
                )
            elif "illegal option" in _output:
                if bias:
                    logging.critical(
                        f"Scanner ({_sdr_name}) - rtl_power reported an illegal option was used. Are you using a rtl_power version with bias tee support?"
                    )
                else:
                    logging.critical(
                       f"Scanner ({_sdr_name}) - rtl_power reported an illegal option was used. (This shouldn't happen... are you running an ancient version?)"
                    )
            else:
                # Something else odd happened, dump the entire error output to the log for further analysis.
                logging.critical(
                    f"Scanner ({_sdr_name}) - rtl_power reported error: {_output}"
                )

            return (None, None, None)

        return read_rtl_power_log(_log_filename, _sdr_name)

    elif sdr_type == "SpyServer":
        # Use a spyserver to obtain power spectral density data

        # Create filename to output to.
        _log_filename = f"log_power_spyserver.csv"

        # If the output log file exists, remove it.
        if os.path.exists(_log_filename):
            os.remove(_log_filename)


        # Add -k 30 option, to SIGKILL rtl_power 30 seconds after the regular timeout expires.
        # Note that this only works with the GNU Coreutils version of Timeout, not the IBM version,
        # which is provided with OSX (Darwin).
        _platform = platform.system()
        if "Darwin" in _platform:
            _timeout_kill = ""
        else:
            _timeout_kill = "-k 30 "

        _timeout_cmd = f"timeout {_timeout_kill}{integration_time+10}"

        _frequency_centre = int(frequency_start + (frequency_stop-frequency_start)/2.0)

        # Note we are using the '-o' option here, which allows us to still get
        # spectrum data even if we have specified a frequency which is out of
        # the range of a locked spyserver.
        _ss_power_cmd = (
            f"{_timeout_cmd} {ss_power_path} "
            f"-f {_frequency_centre} "
            f"-i {integration_time} -1 -o "
            f"-r {sdr_hostname} -q {sdr_port} "
            f"{_log_filename}"
        )

        _sdr_name = get_sdr_name(
            sdr_type=sdr_type,
            rtl_device_idx=rtl_device_idx,
            sdr_hostname=sdr_hostname,
            sdr_port=sdr_port
            )

        logging.info(f"Scanner ({_sdr_name}) - Running frequency scan.")
        logging.debug(
            f"Scanner ({_sdr_name}) - Running command: {_ss_power_cmd}"
        )

        try:
            _output = subprocess.check_output(
                _ss_power_cmd, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            # Something went wrong...
            logging.critical(
                f"Scanner ({_sdr_name}) - ss_power call failed with return code {e.returncode}."
            )
            # Look at the error output in a bit more details.
            _output = e.output.decode("ascii")

            if "outside currently allowed range" in _output:
                logging.critical(
                    f"Scanner ({_sdr_name}) - Centre of scan range ({_frequency_centre} Hz) outside of allowed SpyServer tuning range."
                )
            else:
                logging.critical(
                    f"Scanner ({_sdr_name}) - Other Error: {_output}"
                )

            return (None, None, None)

        return read_rtl_power_log(_log_filename, _sdr_name)

    elif sdr_type == "KA9Q":
        # Use iqrecord to obtain power spectral density data

        _iq_filename = f"log_power_{rtl_device_idx}.iq"
        _log_filename = f"log_power_{rtl_device_idx}.log"

        # Add -k 30 option, to SIGKILL iqrecord 30 seconds after the regular timeout expires.
        # Note that this only works with the GNU Coreutils version of Timeout, not the IBM version,
        # which is provided with OSX (Darwin).
        _platform = platform.system()
        if "Darwin" in _platform:
            _timeout_kill = ""
        else:
            _timeout_kill = "-k 30 "

        _timeout_cmd = f"timeout {_timeout_kill}{integration_time+10}"

        _iq_cmd = (
            f"{_timeout_cmd} {ka9q_iqrecord_path} "
            f"-S sdr1-ctl.local "
            f"-d {integration_time} "
            f"-o {_iq_filename} "
            f"sdr1.local"
        )
        # TODO: make this configurable

        _sdr_name = get_sdr_name(
            sdr_type=sdr_type,
            rtl_device_idx=rtl_device_idx,
            sdr_hostname=sdr_hostname,
            sdr_port=sdr_port
            )

        logging.info(f"Scanner ({_sdr_name}) - Running frequency scan.")
        logging.info(
            f"Scanner ({_sdr_name}) - Running command: {_iq_cmd}"
        )

        try:
            _output = subprocess.check_output(
                _iq_cmd, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            # Something went wrong...
            logging.error(
                f"Scanner ({_sdr_name}) - iqrecod call failed with return code {e.returncode}."
            )
            # Look at the error output in a bit more details.
            _output = e.output.decode("utf-8")
            # Something odd happened, dump the entire error output to the log for further analysis.
            logging.critical(
                f"Scanner ({_sdr_name}) - iqrecod reported error: {_output}"
            )

            return (None, None, None)

        crop = 0.25
        ci = ka9q_get_channel_info()
        freq = ci['freq']
        sample_rate = ci['sample_rate']
        now = datetime.utcnow().strftime('%Y-%m-%d, %H:%M:%S')
        # f"-f {frequency_start}:{frequency_stop}:{step} "
        k = sample_rate // step
        chunk_size = 1
        while chunk_size < k:
            chunk_size *= 2
        data = np.fromfile(_iq_filename, dtype=np.int8)
        data_c = data.astype(np.float32).view(np.complex64) / 127
        power = np.zeros(chunk_size)
        nchunks = len(data_c) // chunk_size
        for n in range(0, nchunks):
            idx = n * chunk_size
            f = fft.fft(data_c[idx:idx+chunk_size])
            power += np.real(np.conj(f) * f)
        power = fft.fftshift(power)
        l = int((len(power) - len(power) * (1-crop)) // 2)
        power = power[l:-l]
        power /= nchunks * chunk_size
        p = 10 * np.log10(power)
        freq_lo = int(freq - sample_rate // 2 * (1-crop))
        freq_hi = int(freq + sample_rate // 2 * (1-crop))

        np.savetxt(
            _log_filename,
            p,
            fmt='%.2f',
            newline=', ',
            header='%s, %d, %d, %.2f, %d' % (now, freq_lo, freq_hi, sample_rate/chunk_size, nchunks,),
            comments=''
        )

        return read_rtl_power_log(_log_filename, _sdr_name)

    else:
        raise NotImplementedError(f"Not implemented for sdr_type={sdr_type}")


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG
    )

    _sdr_type = "RTLSDR"

    #print(f"Test RTLSDR 0: {test_sdr(_sdr_type)}")

    _freq = 401500000
    _sample_rate = 48000
    _fm_bw = 15000
    _device_idx = "00000004"

    print(f"RTLSDR IQ (AGC): {get_sdr_iq_cmd(_sdr_type, _freq, _sample_rate)}")
    print(f"RTLSDR IQ (AGC + Fixed device): {get_sdr_iq_cmd(_sdr_type, _freq, _sample_rate, rtl_device_idx=_device_idx)}")
    print(f"RTLSDR IQ (Fixed Gain): {get_sdr_iq_cmd(_sdr_type, _freq, _sample_rate, gain=30.0, bias=True)}")

    print(f"RTLSDR FM (AGC): {get_sdr_fm_cmd(_sdr_type, _freq, _fm_bw, _sample_rate)}")
    print(f"RTLSDR FM (Fixed Gain): {get_sdr_fm_cmd(_sdr_type, _freq, _fm_bw, _sample_rate, gain=30.0, bias=True, highpass=20)}")

    # (freq, power, step) = get_power_spectrum(
    #     sdr_type="RTLSDR"
    # )

    (freq, power, step) = get_power_spectrum(
        sdr_type="SpyServer",
        sdr_hostname="10.0.0.222",
        sdr_port=5555,
        frequency_start=400100000,
        frequency_stop=404900000
    )
    print(freq)
    print(power)
    print(step)
