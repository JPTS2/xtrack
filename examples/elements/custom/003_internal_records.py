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
# Choose the kicks and recording capacities
################################################################################
P0C             = 1e9            # Reference momentum [eV/c], proton
KICK_A          = 1e-5           # Normalized horizontal momentum increment
KICK_B          = -0.5e-5
NUM_PARTICLES   = 3
NUM_TURNS       = 2
CAPACITY_BEFORE = 16             # One slot per particle and element passage
CAPACITY_AFTER  = 5              # Deliberately too small for all events

################################################################################
# Define two event tables and a logging element
################################################################################

########################################
# Declare the table and nested record schemas
########################################
# _index is the required RecordIndex field of each author-defined table.
class PassageTable(xo.HybridClass):
    _xofields = {
        '_index':      xt.RecordIndex,
        'px':          xo.Float64[:],
        'at_element':  xo.Int64[:],
        'at_turn':     xo.Int64[:],
        'particle_id': xo.Int64[:]}


class KickRecord(xo.HybridClass):
    _xofields = {
        'before': PassageTable,
        'after':  PassageTable}


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
            PassageTableData before = NULL;
            PassageTableData after = NULL;
            if (record) {
                before = KickRecordData_getp_before(record);
                after = KickRecordData_getp_after(record);
            }

            START_PER_PARTICLE_BLOCK(part0, part);
                double px_before = LocalParticle_get_px(part);
                LocalParticle_add_to_px(part, kick);
                if (record) {
                    // Full tables return -1; each table has its own index.
                    int64_t slot = RecordIndex_get_slot(
                        PassageTableData_getp__index(before));
                    if (slot >= 0) {
                        PassageTableData_set_px(before, slot, px_before);
                        PassageTableData_set_at_element(before, slot,
                            LocalParticle_get_at_element(part));
                        PassageTableData_set_at_turn(before, slot,
                            LocalParticle_get_at_turn(part));
                        PassageTableData_set_particle_id(before, slot,
                            LocalParticle_get_particle_id(part));
                    }
                    slot = RecordIndex_get_slot(
                        PassageTableData_getp__index(after));
                    if (slot >= 0) {
                        PassageTableData_set_px(after, slot,
                            LocalParticle_get_px(part));
                        PassageTableData_set_at_element(after, slot,
                            LocalParticle_get_at_element(part));
                        PassageTableData_set_at_turn(after, slot,
                            LocalParticle_get_at_turn(part));
                        PassageTableData_set_particle_id(after, slot,
                            LocalParticle_get_particle_id(part));
                    }
                }
            END_PER_PARTICLE_BLOCK;
        }
    ''']

################################################################################
# Share one nested record between two line elements
################################################################################

########################################
# Allocate each table independently
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
capacities = {'before': CAPACITY_BEFORE, 'after': CAPACITY_AFTER}
record = line.start_internal_logging_for_elements_of_type(
    LoggedKick, capacity = capacities)
line.track(particles, num_turns = NUM_TURNS)

########################################
# Read only the populated slots of each table
########################################
# The index is part of the table schema above. Event order need not follow
# particle IDs on other contexts; compare events by their stored identifiers.
print('Requested events per table:', 2 * NUM_PARTICLES * NUM_TURNS)
for label in ['before', 'after']:
    events       = getattr(record, label)
    num_recorded = events._index.num_recorded
    print(label, 'recorded:', num_recorded, '/ capacity:', capacities[label])
    xt.Table(
        data    = {
            'event':       np.arange(num_recorded),
            'particle_id': events.particle_id[:num_recorded],
            'at_turn':     events.at_turn[:num_recorded],
            'at_element':  events.at_element[:num_recorded],
            'px':          events.px[:num_recorded]},
        index   = 'event').show()
print('Final px despite the full after table:', particles.px)
print('Expected final px:', NUM_TURNS * (KICK_A + KICK_B))

########################################
# Stop logging while continuing to track
########################################
line.stop_internal_logging_for_elements_of_type(LoggedKick)
px_before = particles.px.copy()
line.track(particles, num_turns = 1)
print('Retained before-table length:', record.before._index.num_recorded)
print('Additional px after stopping:', particles.px - px_before)
print('Expected additional px:', KICK_A + KICK_B)

################################################################################
# Attach a record to selected standalone elements
################################################################################

########################################
# Share a separate IO buffer and record
########################################
standalone_a = LoggedKick(_context = context, kick_px = KICK_A)
standalone_b = LoggedKick(_context = context, kick_px = KICK_B)
io_buffer = xt.new_io_buffer(_context = context)
xt.start_internal_logging(
    elements    = [standalone_a, standalone_b],
    io_buffer   = io_buffer,
    capacity    = capacities)
standalone_record = standalone_a.record
standalone_particles = xt.Particles(
    _context    = context,
    p0c         = P0C,
    x           = np.zeros(NUM_PARTICLES))
standalone_a.track(standalone_particles)
standalone_b.track(standalone_particles)
print('Both elements use the same record:',
      standalone_a.record is standalone_b.record)

########################################
# Stop only one element and retain both tables
########################################
xt.stop_internal_logging(elements = standalone_a)
standalone_a.track(standalone_particles)
standalone_b.track(standalone_particles)
print('Before-table events after selective stop:',
      standalone_record.before._index.num_recorded)
print('Expected events from a, b, then only b:', 3 * NUM_PARTICLES)
xt.stop_internal_logging(elements = standalone_b)
px_before = standalone_particles.px.copy()
standalone_a.track(standalone_particles)
standalone_b.track(standalone_particles)
print('Retained standalone before-table length:',
      standalone_record.before._index.num_recorded)
print('Additional px with both loggers stopped:',
      standalone_particles.px - px_before)
