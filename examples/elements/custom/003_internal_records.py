# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# Choose the kicks and recording capacity
################################################################################
P0C             = 1e9            # Reference momentum [eV/c], proton
KICK_A          = 1e-5           # Normalized horizontal momentum increment
KICK_B          = -0.5e-5
NUM_PARTICLES   = 3
NUM_TURNS       = 2
CAPACITY        = 16             # One slot per particle and element passage

################################################################################
# Define an event record and a logging element
################################################################################

########################################
# Declare the shared record schema
########################################
class KickRecord(xo.HybridClass):
    _xofields = {
        '_index':      xt.RecordIndex,
        'kick_px':     xo.Float64[:],
        'at_element':  xo.Int64[:],
        'at_turn':     xo.Int64[:],
        'particle_id': xo.Int64[:]}


########################################
# Apply each kick independently of logging
########################################
class LoggedKick(xt.BeamElement):
    _xofields               = {'kick_px': xo.Float64}
    _internal_record_class  = KickRecord
    _extra_c_sources        = [r'''
        #include "xtrack/headers/track.h"

        GPUFUN
        void LoggedKick_track_local_particle(
                LoggedKickData el, LocalParticle* part0) {
            double kick = LoggedKickData_get_kick_px(el);
            KickRecordData record =
                LoggedKickData_getp_internal_record(el, part0);
            RecordIndex index = NULL;
            if (record) index = KickRecordData_getp__index(record);

            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_add_to_px(part, kick);
                if (record) {
                    // Slot allocation is thread safe; full records return -1.
                    int64_t slot = RecordIndex_get_slot(index);
                    if (slot >= 0) {
                        KickRecordData_set_kick_px(record, slot, kick);
                        KickRecordData_set_at_element(record, slot,
                            LocalParticle_get_at_element(part));
                        KickRecordData_set_at_turn(record, slot,
                            LocalParticle_get_at_turn(part));
                        KickRecordData_set_particle_id(record, slot,
                            LocalParticle_get_particle_id(part));
                    }
                }
            END_PER_PARTICLE_BLOCK;
        }
    ''']

################################################################################
# Record passages through two elements of the same type
################################################################################

########################################
# Attach one record to both elements
########################################
context = xo.ContextCpu()
context.allow_kernel_compilation = True
line = xt.Line(
    elements        = {
        'kick_a': LoggedKick(_context = context, kick_px = KICK_A),
        'kick_b': LoggedKick(_context = context, kick_px = KICK_B)})
line.build_tracker(_context = context)
particles = xt.Particles(
    _context    = context,
    p0c         = P0C,
    x           = np.zeros(NUM_PARTICLES),
    particle_id = 10 + np.arange(NUM_PARTICLES))
record = line.start_internal_logging_for_elements_of_type(
    LoggedKick, capacity = CAPACITY)
line.track(particles, num_turns = NUM_TURNS)

########################################
# Read only the populated slots
########################################
# The record belongs to this logging session and is shared by both instances.
# Event order need not follow particle ID order on other execution contexts.
num_recorded = record._index.num_recorded
print('Requested events:', 2 * NUM_PARTICLES * NUM_TURNS)
print('Recorded events:', num_recorded, '/ capacity:', CAPACITY)
xt.Table(
    data    = {
        'event':       np.arange(num_recorded),
        'particle_id': record.particle_id[:num_recorded],
        'at_turn':     record.at_turn[:num_recorded],
        'at_element':  record.at_element[:num_recorded],
        'kick_px':     record.kick_px[:num_recorded]},
    index   = 'event').show()

################################################################################
# Stop logging while continuing to track
################################################################################
line.stop_internal_logging_for_elements_of_type(LoggedKick)
px_before = particles.px.copy()
line.track(particles, num_turns = 1)
print('Record length after stopping:', record._index.num_recorded)
print('Additional px after stopping:', particles.px - px_before)
print('Expected additional px:', KICK_A + KICK_B)
