; RUN: opt -mtriple=amdgpu-- -passes=loop-unroll -S %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -mtriple=amdgpu-- -passes=loop-unroll -amdgpu-unroll-runtime-local=false -S %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,LOCALOFF

; Runtime unrolling of loops that access LDS.
;
; AMDGPU enables runtime unrolling for every loop (see getUnrollingPreferences),
; and -amdgpu-unroll-runtime-local=false opts loops that access LDS back out.
; That flag must apply to every LDS loop, not only to the ones admitted by the
; LDS combining heuristics that follow it (single LDS GEP per block, nest depth
; <= 2, pointer operand a global or an argument).
;
; Regression guard: when the flag was applied after those heuristics, it could
; only ever disable the loops they admitted, while the loops they rejected kept
; the default of true - the inverse of what the flag says it does. @deep_nest
; below is exactly such a rejected loop (nest depth 3).

@lds = internal addrspace(3) global [1024 x float] poison, align 4

; A shallow, single-GEP LDS loop: admitted by the heuristics.
; Runtime unrolled by default; the flag must turn that off.
define void @lds_shallow(ptr addrspace(1) %out, i32 %n) {
; CHECK-LABEL: @lds_shallow(
; DEFAULT: %xtraiter = and i32 %n,
; DEFAULT: for.body.epil:
; LOCALOFF-NOT: xtraiter
; LOCALOFF-NOT: epil
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %idx = zext i32 %iv to i64
  %ptr = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i64 %idx
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %res.shallow = phi float [ 0.0, %entry ], [ %add, %for.body ]
  store float %res.shallow, ptr addrspace(1) %out, align 4
  ret void
}

; The innermost loop of a 3-deep nest. getLoopDepth() > 2, so the LDS combining
; heuristics reject it and take the `continue`.
;
; By default it is runtime unrolled, because AMDGPU enables runtime unrolling
; globally. With the flag off it must not be unrolled - before the fix it still
; was, because the rejected path never reached the flag.
define void @deep_nest(ptr addrspace(1) %out, i32 %n1, i32 %n2, i32 %n3) {
; The negative check keys on the inner loop's TRIP COUNT (%n3), which is the only
; reliable signal here. Two things make the obvious alternatives wrong:
;
;  - a bare "NOT: xtraiter" fails, because with the flag off the inner loop is
;    suppressed and the unroller unrolls %mid.body instead. %mid.body holds no
;    LDS GEP of its own - the GEP is in the subloop's block, which is skipped
;    when the enclosing loop is analysed - so it keeps the global UP.Runtime.
;  - a "NOT: inner.body.epil" fails for a subtler reason: unrolling %mid.body
;    clones the whole inner loop into mid's epilogue, and the clone is *named*
;    inner.body.epil. That name is a copy of the inner loop, not an epilogue of
;    it, so matching on it would assert the opposite of what it appears to.
; CHECK-LABEL: @deep_nest(
; DEFAULT: %xtraiter = and i32 %n3,
; DEFAULT: inner.body.epil:
; LOCALOFF-NOT: and i32 %n3,
entry:
  br label %outer.body

outer.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc.outer = phi float [ 0.0, %entry ], [ %acc.mid.out, %outer.latch ]
  br label %mid.body

mid.body:
  %j = phi i32 [ 0, %outer.body ], [ %j.next, %mid.latch ]
  %acc.mid = phi float [ %acc.outer, %outer.body ], [ %acc.inner.out, %mid.latch ]
  br label %inner.body

inner.body:
  %k = phi i32 [ 0, %mid.body ], [ %k.next, %inner.body ]
  %acc.inner = phi float [ %acc.mid, %mid.body ], [ %add, %inner.body ]
  %sum = add i32 %i, %j
  %sum2 = add i32 %sum, %k
  %off = and i32 %sum2, 1023
  %idx = zext i32 %off to i64
  %ptr = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i64 %idx
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc.inner, %val
  %k.next = add nuw nsw i32 %k, 1
  %inner.cond = icmp eq i32 %k.next, %n3
  br i1 %inner.cond, label %mid.latch, label %inner.body

mid.latch:
  %acc.inner.out = phi float [ %add, %inner.body ]
  %j.next = add nuw nsw i32 %j, 1
  %mid.cond = icmp eq i32 %j.next, %n2
  br i1 %mid.cond, label %outer.latch, label %mid.body

outer.latch:
  %acc.mid.out = phi float [ %acc.inner.out, %mid.latch ]
  %i.next = add nuw nsw i32 %i, 1
  %outer.cond = icmp eq i32 %i.next, %n1
  br i1 %outer.cond, label %exit, label %outer.body

exit:
  %res = phi float [ %acc.mid.out, %outer.latch ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}
