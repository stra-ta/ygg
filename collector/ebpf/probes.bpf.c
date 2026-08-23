// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_CPUS 256
#define EVENT_RING_SIZE (1 << 20)  // 1M events per CPU

struct event {
    __u64 timestamp_ns;
    __u32 cpu;
    __u32 pid;
    __u32 tid;
    __u16 kind;
    __u16 padding;
    __u64 arg0;
    __u64 arg1;
    __u64 arg2;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, EVENT_RING_SIZE * sizeof(struct event));
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} tsc_freq_hz SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} monotonic_offset_ns SEC(".maps");

static __always_inline __u64 get_timestamp_ns(void) {
    __u64 tsc = bpf_ktime_get_ns();
    return tsc;
}

static __always_inline void emit_event(__u16 kind, __u64 arg0, __u64 arg1, __u64 arg2) {
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return;

    e->timestamp_ns = get_timestamp_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    e->kind = kind;
    e->padding = 0;
    e->arg0 = arg0;
    e->arg1 = arg1;
    e->arg2 = arg2;

    bpf_ringbuf_submit(e, 0);
}

// Scheduler events
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next) {
    __u32 prev_tid = BPF_CORE_READ(prev, pid);
    __u32 next_tid = BPF_CORE_READ(next, pid);
    __u64 prev_state = BPF_CORE_READ(prev, __state);
    emit_event(3000, prev_tid, next_tid, prev_state);
    return 0;
}

SEC("tp_btf/sched_wakeup")
int BPF_PROG(sched_wakeup, struct task_struct *p) {
    __u32 target_tid = BPF_CORE_READ(p, pid);
    __u32 target_cpu = BPF_CORE_READ(p, cpu);
    emit_event(3001, target_tid, target_cpu, 0);
    return 0;
}

SEC("tp_btf/sched_migrate_task")
int BPF_PROG(sched_migrate_task, struct task_struct *p, int dest_cpu) {
    __u32 tid = BPF_CORE_READ(p, pid);
    __u32 from_cpu = bpf_get_smp_processor_id();
    emit_event(3002, tid, from_cpu, dest_cpu);
    return 0;
}

// Syscall events
SEC("tp_btf/sys_enter")
int BPF_PROG(sys_enter, struct pt_regs *regs, long id) {
    __u64 arg0 = BPF_CORE_READ(regs, di);
    __u64 arg1 = BPF_CORE_READ(regs, si);
    emit_event(2000, id, arg0, arg1);
    return 0;
}

SEC("tp_btf/sys_exit")
int BPF_PROG(sys_exit, struct pt_regs *regs, long ret) {
    long id = BPF_CORE_READ(regs, orig_ax);
    emit_event(2001, id, ret, 0);
    return 0;
}

// Block I/O events
SEC("tp_btf/block_rq_issue")
int BPF_PROG(block_rq_issue, struct request *rq) {
    __u64 sector = BPF_CORE_READ(rq, __sector);
    __u64 bytes = BPF_CORE_READ(rq, __data_len);
    __u64 rw_flag = BPF_CORE_READ(rq, cmd_flags);
    emit_event(4000, sector, bytes, rw_flag);
    return 0;
}

SEC("tp_btf/block_rq_complete")
int BPF_PROG(block_rq_complete, struct request *rq, blk_status_t error, unsigned int bytes) {
    __u64 sector = BPF_CORE_READ(rq, __sector);
    __u64 latency_ns = bpf_ktime_get_ns() - BPF_CORE_READ(rq, start_time_ns);
    emit_event(4001, sector, bytes, latency_ns);
    return 0;
}

// Network events
SEC("kprobe/tcp_sendmsg")
int BPF_KPROBE(tcp_sendmsg, struct sock *sk, struct msghdr *msg, size_t size) {
    __u32 fd = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    emit_event(5000, size, fd, 0);
    return 0;
}

SEC("kretprobe/tcp_recvmsg")
int BPF_KRETPROBE(tcp_recvmsg, int ret) {
    if (ret > 0) {
        __u32 fd = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
        emit_event(5001, ret, fd, 0);
    }
    return 0;
}

// Page fault events
SEC("tp_btf/page_fault_user")
int BPF_PROG(page_fault_user, unsigned long address, unsigned long error_code) {
    emit_event(6000, address, error_code, 0);
    return 0;
}

SEC("tp_btf/page_fault_kernel")
int BPF_PROG(page_fault_kernel, unsigned long address, unsigned long error_code) {
    emit_event(6001, address, error_code, 0);
    return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";